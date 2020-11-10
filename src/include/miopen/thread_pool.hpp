/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MIOPEN_GUARD_THREAD_POOL_HPP
#define MIOPEN_GUARD_THREAD_POOL_HPP

#include "env.hpp"
#include "logger.hpp"
#include <vector>
#include <queue>
#include <atomic>
#include <future>
#include <condition_variable>
#include <thread>
#include <functional>
#include <stdexcept>

namespace miopen {

//线程池,可以提交变参函数或拉姆达表达式的匿名函数执行,可以获取执行返回值
class ThreadPool
{
    using Task = std::function<void()>; //定义类型
    std::shared_ptr<std::thread> manager_thread;
    std::vector<std::thread> threads;     //线程池
    std::queue<Task> cached_tasks;
    std::queue<Task> wait_execute_tasks;            //任务队列
    std::mutex manager_lock;
    std::mutex execute_tasks_lock;                   //同步
    std::condition_variable manager_cv;   //条件阻塞
    std::condition_variable execute_tasks_cv;
    std::atomic<bool> runing{true};       //线程池是否执行
    std::atomic<int> idle_thread_number{0};     //空闲线程数量
    int wait_count = 0;
    unsigned long int max_thread_number;
    std::atomic<bool> auto_increase_thread{false};

    public:
    inline ThreadPool(unsigned long int size = 4, unsigned long int maxSize = 16)
        : max_thread_number(maxSize)
    {
        AddThread(size);
        manager_thread = std::make_shared<std::thread>(
            [this]{
            while(runing)
            {
                std::unique_lock<std::mutex> lock{manager_lock};
                manager_cv.wait(
                    lock, [this] { return !runing || (idle_thread_number.load() && !wait_execute_tasks.empty())}); // wait 直到有 task
                if(!runing && wait_execute_tasks.empty())
                    return;
                execute_tasks_cv.notify_one();
            }
        });
    }
    inline ~ThreadPool()
    {
        runing.store(false);
        execute_tasks_cv.notify_all(); // 唤醒所有线程执行
        for(std::thread& thr : threads)
        {
            // thread.detach(); // 让线程“自生自灭”
            if(thr.joinable())
                thr.join(); // 等待任务结束， 前提：线程一定会执行完
        }
    }

    public:
    // 提交一个任务
    // 调用.get()获取返回值会等待任务执行完,获取返回值
    // 有两种方法可以实现调用类成员，
    // 一种是使用   bind： .Commit(std::bind(&Dog::sayHello, &dog));
    // 一种是用   mem_fn： .Commit(std::mem_fn(&Dog::sayHello), this)
    template <class F, class... Args>
    auto Commit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>
    {
        if(!runing) // stoped ??
                  // 多个线程中提交，可能存在某个线程已经通过，另外的线程却stoped，导致问题。
            throw std::runtime_error("Commit on ThreadPool is stopped.");

        using RetType =
            decltype(f(args...)); // typename std::result_of<F(Args...)>::type, 函数 f 的返回值类型
        auto task = std::make_shared<std::packaged_task<RetType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)); // 把函数入口及参数,打包(绑定)
        std::future<RetType> future = task->get_future();
        { // 添加任务到队列
            std::lock_guard<std::mutex> lock{
                execute_tasks_lock}; //对当前块的语句加锁  lock_guard 是 mutex 的 stack 封装类，构造的时候
                        //lock()，析构的时候 unlock()
            wait_execute_tasks.emplace([task]() { // push(Task{...}) 放到队列后面
                (*task)();
            });
        }
        if(idle_thread_number < 1 && auto_increase_thread && threads.size() < max_thread_number)
            AddThread(1);
        manager_cv.notify_one(); // 唤醒一个线程执行
        
        return future;
    }

    //空闲线程数量
    int IdleCount() { return idle_thread_number; }
    //设置线程池大小是否随着任务量自动增长
    void AutoIncrease(bool bAutoSize = false) { auto_increase_thread.store(bAutoSize); }
    void AddThreads(unsigned short size)
    {
        if(!runing) // stoped ??
            throw std::runtime_error("ThreadPool is stopped.");
        if(auto_increase_thread)
            AddThread(size);
        else
            throw std::runtime_error("Forbidden.");
    }
    //线程数量
    int PoolSize() { return threads.size(); }

    private:
    //添加指定数量的线程
    void AddThread(unsigned short size)
    {
        for(; threads.size() < max_thread_number && size > 0; --size)
        { //增加线程数量,但不超过 预定义数量 THREADPOOL_MAX_NUM
            threads.emplace_back([this] { //工作线程函数
                while(runing)
                {
                    Task task; // 获取一个待执行的 task
                    {
                        // unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
                        std::unique_lock<std::mutex> lock{execute_tasks_lock};
                        execute_tasks_cv.wait(
                            lock, [this] { return !runing || !wait_execute_tasks.empty(); }); // wait 直到有 task
                        if(!runing && wait_execute_tasks.empty())
                            return;
                        task = std::move(wait_execute_tasks.front()); // 按先进先出从队列取一个 task
                        wait_execute_tasks.pop();
                    }
                    idle_thread_number--;
                    task(); //执行任务
                    idle_thread_number++;
                    manager_cv.notify_one();
                }
            });
            idle_thread_number++;
        }
    }
};

} // namespace miopen

#endif