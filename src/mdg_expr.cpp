#include <miopen/mdg_expr.hpp>

namespace miopen {

MDGExprParser::MDGExprParser() : MDGExprParser::base_type(expression)
{
    using qi::_1;
    using qi::_2;
    using qi::_val;
    using qi::alnum;
    using qi::alpha;
    using qi::char_;
    using qi::double_;
    using qi::hex;
    using qi::int_;
    using qi::lexeme;
    // clang-format off
    expression =
        additive_expr                        [ _val = _1]
        ;

    additive_expr =
        primary_expr                         [ _val = _1 ]
        >> *(ops >> primary_expr)  [ _val = makebinary(_1, _val, _2)]
        ;

    primary_expr =
        // cppcheck-suppress compareBoolExpressionWithInt
        ( '(' > expression > ')' )         [ _val = _1 ]
        | constant                           [ _val = _1 ]
        | variable                           [ _val = _1 ]
        ;
    ops =     
         (char_(">") >> char_("="))                                                                                                             
        | (char_("<") >> char_("="))                                                                                                            
        | (char_("!") > char_("="))                                                                                                             
        | (char_("=") >> char_("=") >> char_("="))                                                                                              
        | (char_("=") >> char_("=") )                                                                                                           
        | char_("-+*/^&|~><%")                                                                                                                   
        ;  
    constant = lexeme ["0x" >> hex] | int_ | double_ ;
    variable = lexeme [ +(alpha >> *( alnum | char_('_'))) ];
    // clang-format on

    BOOST_SPIRIT_DEBUG_NODE(expression);
    BOOST_SPIRIT_DEBUG_NODE(additive_expr);

    BOOST_SPIRIT_DEBUG_NODE(primary_expr);
    BOOST_SPIRIT_DEBUG_NODE(constant);
    BOOST_SPIRIT_DEBUG_NODE(variable);
}

} // namespace miopen
