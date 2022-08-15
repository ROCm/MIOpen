/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
