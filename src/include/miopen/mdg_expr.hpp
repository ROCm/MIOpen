#ifndef MIOPEN_MDG_EXPR_H
#define MIOPEN_MDG_EXPR_H

// #define BOOST_SPIRIT_DEBUG

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>

#include <cassert>
#include <memory>
#include <iostream>
#include <unordered_map>
// Workaround tidy issues when using BOOST_FOREACH
#ifdef MIOPEN_USE_CLANG_TIDY
#define BOOST_FOREACH(x, y) for(x : y) // NOLINT
#endif
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/support_utree.hpp>
#include <miopen/fusion_ops.hpp>

namespace miopen {

namespace qi     = boost::spirit::qi;
namespace ascii  = boost::spirit::ascii;
namespace phx    = boost::phoenix;
namespace spirit = boost::spirit;

using Iterator = std::string::const_iterator;

struct MDGExprParser : qi::grammar<Iterator, spirit::utree(), ascii::space_type>
{
    struct MakeBinaryExpression
    {
        spirit::utree operator()(spirit::utf8_symbol_type op,
                                 spirit::utree const& lhs,
                                 spirit::utree const& rhs) const
        {
            spirit::utree expr;
            expr.push_back(op);
            expr.push_back(lhs);
            expr.push_back(rhs);
            return expr;
        }
    };

    phx::function<MakeBinaryExpression> makebinary;

    MDGExprParser();
    qi::rule<Iterator, spirit::utree(), ascii::space_type> expression;
    qi::rule<Iterator, spirit::utree(), ascii::space_type> additive_expr;
    qi::rule<Iterator, spirit::utree(), ascii::space_type> primary_expr;
    qi::rule<Iterator, spirit::utree(), ascii::space_type> constant;
    qi::rule<Iterator, spirit::utf8_symbol_type(), ascii::space_type> ops;
    qi::rule<Iterator, std::string(), ascii::space_type> variable;
};

struct visit_res
{
    int res         = 0;
    bool b_res      = false;
    MDGraph_op_t op = OpAny;
    std::string sym;
    std::unordered_map<std::string, int> tabl;
};

struct tree_visit
{
    std::unordered_map<std::string, int> tabl;
    std::function<bool(const std::string&, int&)> var_lookup;

    using result_type = visit_res;
    tree_visit(){};
    tree_visit(std::function<bool(const std::string&, int&)> f) : var_lookup(f){};
    tree_visit(std::function<bool(const std::string&, int&)> f,
               std::unordered_map<std::string, int> t)
        : tabl(t), var_lookup(f){};
    visit_res operator()(spirit::utree::invalid_type) const { return {}; }

    visit_res operator()(spirit::utree::nil_type) const { return {}; };

    visit_res operator()(double d)
    {
        visit_res r;
        r.res = static_cast<int>(d);
        return r;
    }

    visit_res operator()(int i)
    {
        visit_res r;
        r.res = i;
        return r;
    }

    template <typename T>
    visit_res operator()(T /*val*/)
    {
        return {};
    }

    visit_res operator()(bool b)
    {
        visit_res r;
        r.b_res = b;
        return r;
    }

    visit_res operator()(spirit::binary_range_type const& /*b*/) const { return {}; }

    visit_res operator()(spirit::utf8_string_range_type const& str)
    {
        visit_res r;
        using iterator = spirit::utf8_string_range_type::const_iterator;
        iterator i     = str.begin();
        std::string sym(i, str.end());
        int v = 0;
        if(var_lookup(sym, v))
        {
            r.res = v;
        }
        else
        {
            if(tabl.count(sym) == 1)
            {
                r.res = tabl.at(sym);
            }
            else
                r.sym = sym;
        }
        return r;
    }

    visit_res operator()(spirit::utf8_symbol_range_type const& str)
    {
        using iterator = spirit::utf8_symbol_range_type::const_iterator;
        iterator i     = str.begin();
        std::string sym(i, str.end());
        // This should return a function instead of a data struct
        visit_res r;
        if(sym == "+")
            r.op = OpAdd;
        else if(sym == "-")
            r.op = OpSub;
        else if(sym == "*")
            r.op = OpMul;
        else if(sym == "/")
            r.op = OpDiv;
        else if(sym == "%")
            r.op = OpModulo;
        else if(sym == ">=")
            r.op = OpGTE;
        else if(sym == "<=")
            r.op = OpLTE;
        else if(sym == "====")
            r.op = OpEqual;
        else if(sym == "!=")
            r.op = OpNotEqual;
        else if(sym == "^")
            r.op = OpPow;
        else if(sym == "&")
            r.op = OpAnd;
        else if(sym == "|")
            r.op = OpOr;
        else if(sym == "~")
            r.op = OpCeil;
        else if(sym == "===")
            r.op = OpAssign;
        else if(sym == ">>")
            r.op = OpGT;
        else if(sym == "<<")
            r.op = OpLT;
        else
        {
            MIOPEN_THROW(miopenStatusInternalError, "Parsing error: Unknown operator: " + sym);
        }
        return r;
    }

    template <typename Iterator>
    visit_res operator()(boost::iterator_range<Iterator> const& range)
    {
        std::vector<spirit::utree> v(range.begin(), range.end());
        assert(v.size() == 3);
        visit_res op_res, lhs_res, rhs_res, r;
        op_res  = boost::spirit::utree::visit(v[0], *this);
        lhs_res = boost::spirit::utree::visit(v[1], *this);
        rhs_res = boost::spirit::utree::visit(v[2], *this);

        if(op_res.op != OpAssign && !lhs_res.sym.empty())
        {
            std::string sym = lhs_res.sym;
            MIOPEN_THROW("Invalid variable access: " + sym);
        }

        switch(op_res.op)
        {
        // Arith ops
        case OpAdd: r.res = lhs_res.res + rhs_res.res; break;
        case OpSub: r.res = lhs_res.res - rhs_res.res; break;
        case OpMul: r.res = lhs_res.res * rhs_res.res; break;
        case OpDiv: r.res = lhs_res.res / rhs_res.res; break;
        case OpModulo: r.res = lhs_res.res % rhs_res.res; break;
        case OpPow: r.res = static_cast<int>(std::pow(lhs_res.res, rhs_res.res)); break;
        case OpCeil: {
            int vv = lhs_res.res;
            int mm = rhs_res.res;
            r.res  = (vv % mm != 0) ? (vv / mm + 1) * mm : vv;
            break;
        }
        case OpAssign: {
            int val = 0;
            if(var_lookup(lhs_res.sym, val))
                MIOPEN_THROW("Invalid variable assignment: " + lhs_res.sym);
            MIOPEN_LOG_I2(" Adding variable: " + lhs_res.sym);
            r.tabl[lhs_res.sym] = rhs_res.res;
            r.b_res             = true;
            break;
        }
        // Logical ops
        case OpEqual:
            r.b_res = lhs_res.res == rhs_res.res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpNotEqual:
            r.b_res = lhs_res.res != rhs_res.res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpGTE:
            r.b_res = lhs_res.res >= rhs_res.res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpLTE:
            r.b_res = lhs_res.res <= rhs_res.res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpGT:
            r.b_res = lhs_res.res > rhs_res.res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpLT:
            r.b_res = lhs_res.res < rhs_res.res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpAnd:
            r.b_res = lhs_res.b_res && rhs_res.b_res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpOr:
            r.b_res = lhs_res.b_res || rhs_res.b_res;
            r.res   = static_cast<int>(r.b_res);
            break;
        case OpAny:
        case OpEval: MIOPEN_THROW("Unsupported op");
        }
        return r;
    }

    visit_res operator()(spirit::any_ptr const&) const { return {}; }

    visit_res operator()(spirit::function_base const&) const { return {}; }
};
} // namespace miopen

#endif
