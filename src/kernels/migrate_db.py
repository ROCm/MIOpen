import sqlite3
import sys
arch = sys.argv[1]
num_cu = int(sys.argv[2])

db_filename = arch + '.db'

conn = sqlite3.connect(db_filename)
cur = conn.cursor()

slv_cnt = {}
for cnt, slv in cur.execute("select count(*), solver from perf_db where arch = ? and num_cu = ? group by solver;", (arch, num_cu)):
	slv_cnt[slv] = cnt

print('Solvers: ')
print(slv_cnt)
print('Solvers End')

total_cnt = cur.execute("SELECT count(*) from perf_db where arch = ? and num_cu = ?", (arch, num_cu)).fetchone()[0]

cur.execute("DELETE FROM perf_db where arch != ?", (arch, ))
cur.execute("DELETE FROM perf_db where num_cu != ?", (num_cu, ))
res = cur.execute("SELECT DISTINCT arch, num_cu from perf_db;").fetchall()
assert(len(res) == 1)
a, n = res[0]
assert(a == arch)
assert(n == num_cu)

cur.execute("drop index idx_perf_db;")
cur.execute("CREATE TEMPORARY TABLE `perf_db_bkup` (`id` INTEGER PRIMARY KEY ASC,`solver` TEXT NOT NULL,`config` INTEGER NOT NULL,`params` TEXT NOT NULL);")
cur.execute("INSERT INTO perf_db_bkup(id, solver, config, params) SELECT id, solver, config, params from perf_db;")
cur.execute("drop table perf_db;")
cur.execute("CREATE TABLE `perf_db` (`id` INTEGER PRIMARY KEY ASC,`solver` TEXT NOT NULL,`config` INTEGER NOT NULL,`params` TEXT NOT NULL);")
cur.execute("INSERT INTO perf_db SELECT id, solver, config, params from perf_db_bkup;")
cur.execute("DROP Table perf_db_bkup;")
cur.execute("CREATE UNIQUE INDEX `idx_perf_db` ON perf_db(solver, config);")
orphan_cnt = cur.execute("SELECT count(*) from config where id not in ( select config from perf_db);").fetchone()[0]
if orphan_cnt != 0:
	cur.execute("Delete from config where id not in (select config from perf_db);")


new_cnt = cur.execute("SELECT count(*) from perf_db").fetchone()[0]
assert(new_cnt == total_cnt)
new_slv_cnt = {}
for cnt, slv in cur.execute("select count(*), solver from perf_db group by solver;"):
	new_slv_cnt[slv] = cnt

for slv, cnt in slv_cnt.items():
	assert(slv in new_slv_cnt.keys())
	assert(cnt == new_slv_cnt[slv])
conn.commit()
conn.close()
