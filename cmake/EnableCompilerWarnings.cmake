# - Enable warning all for gcc/clang or use /W4 for visual studio

## Strict warning level
if (MSVC)
    # Use the highest warning level for visual studio.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")
    # set(CMAKE_CXX_WARNING_LEVEL 4)
    # if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    #     string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    # else ()
    #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    # endif ()

    # set(CMAKE_C_WARNING_LEVEL 4)
    # if (CMAKE_C_FLAGS MATCHES "/W[0-4]")
    #     string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    # else ()
    #     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    # endif ()

else()
    foreach(COMPILER C CXX)
        set(CMAKE_COMPILER_WARNINGS)
        # use -Wall for gcc and clang
        list(APPEND CMAKE_COMPILER_WARNINGS 
            -Wall
            -Wextra
            -Wcomment
            -Wendif-labels
            -Wformat
            -Winit-self
            -Wreturn-type
            -Wsequence-point
            -Wshadow
            -Wswitch
            -Wtrigraphs
            -Wundef
            -Wuninitialized
            -Wunreachable-code
            -Wunused

            -Wno-sign-compare
        )
        if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang")
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Wabi
                -Wabstract-final-class
                -Waddress
                -Waddress-of-array-temporary
                -Waddress-of-temporary
                -Waggregate-return
                -Wambiguous-macro
                -Wambiguous-member-template
                -Wanonymous-pack-parens
                -Warray-bounds
                -Warray-bounds-pointer-arithmetic
                -Wassign-enum
                -Watomic-properties
                -Wattributes
                -Wavailability
                -Wbackslash-newline-escape
                # -Wbad-array-new-length
                -Wbad-function-cast
                -Wbind-to-temporary-copy
                -Wbuiltin-macro-redefined
                -Wbuiltin-requires-header
                -Wcast-align
                -Wcast-qual
                -Wchar-align
                -Wcomments
                -Wcompare-distinct-pointer-types
                -Wconditional-type-mismatch
                -Wconditional-uninitialized
                -Wconfig-macros
                -Wconstant-logical-operand
                -Wconstexpr-not-const
                -Wconversion-null
                -Wcovered-switch-default
                -Wctor-dtor-privacy
                -Wdangling-field
                -Wdangling-initializer-list
                -Wdelete-incomplete
                -Wdeprecated
                -Wdivision-by-zero
                -Wduplicate-decl-specifier
                -Wduplicate-enum
                -Wduplicate-method-arg
                -Wduplicate-method-match
                -Wdynamic-class-memaccess
                -Wempty-body
                -Wenum-compare
                -Wexplicit-ownership-type
                -Wextern-initializer
                -Wfloat-equal
                -Wgnu-array-member-paren-init
                -Wheader-guard
                -Wheader-hygiene
                -Widiomatic-parentheses
                -Wignored-attributes
                -Wimplicit-conversion-floating-point-to-bool
                -Wimplicit-exception-spec-mismatch
                -Wimplicit-fallthrough
                -Wincompatible-library-redeclaration
                -Wincompatible-pointer-types
                # -Winherited-variadic-ctor
                -Winline
                -Wint-conversions
                -Wint-to-pointer-cast
                -Winteger-overflow
                -Winvalid-constexpr
                -Winvalid-noreturn
                -Winvalid-offsetof
                -Winvalid-pch
                -Winvalid-pp-token
                -Winvalid-source-encoding
                -Winvalid-token-paste
                -Wloop-analysis
                -Wmain
                -Wmain-return-type
                -Wmalformed-warning-check
                -Wmethod-signatures
                -Wmismatched-parameter-types
                -Wmismatched-return-types
                -Wmissing-declarations
                -Wmissing-format-attribute
                -Wmissing-include-dirs
                -Wmissing-sysroot
                -Wmissing-variable-declarations
                -Wnarrowing
                -Wnested-externs
                -Wnewline-eof
                -Wnon-pod-varargs
                -Wnon-virtual-dtor
                -Wnull-arithmetic
                -Wnull-character
                -Wnull-dereference
                -Wodr
                -Wold-style-definition
                -Wout-of-line-declaration
                -Wover-aligned
                -Woverflow
                -Woverriding-method-mismatch
                -Wpacked
                -Wpointer-sign
                -Wpointer-to-int-cast
                -Wpointer-type-mismatch
                -Wpredefined-identifier-outside-function
                -Wredundant-decls
                -Wreinterpret-base-class
                -Wreserved-user-defined-literal
                -Wreturn-stack-address
                -Wsection
                -Wserialized-diagnostics
                -Wshift-count-negative
                -Wshift-count-overflow
                -Wshift-overflow
                -Wshift-sign-overflow
                # -Wsign-compare
                -Wsign-promo
                -Wsizeof-pointer-memaccess
                -Wstack-protector
                -Wstatic-float-init
                -Wstring-compare
                -Wstrlcpy-strlcat-size
                -Wstrncat-size
                -Wswitch-default
                -Wswitch-enum
                -Wsynth
                -Wtautological-compare
                -Wtentative-definition-incomplete-type
                -Wthread-safety
                -Wtype-limits
                -Wtype-safety
                -Wtypedef-redefinition
                -Wtypename-missing
                -Wundefined-inline
                -Wundefined-reinterpret-cast
                -Wunicode
                -Wunicode-whitespace
                -Wunused-exception-parameter
                -Wunused-macros
                -Wunused-member-function
                -Wunused-parameter
                -Wunused-volatile-lvalue
                -Wused-but-marked-unused
                -Wuser-defined-literals
                -Wvarargs
                -Wvector-conversions
                -Wvexing-parse
                -Wvisibility
                -Wvla
                -Wweak-template-vtables
                # -Wweak-vtables
                -Wwrite-strings
            )
        else()
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Wno-missing-field-initializers
                -Wno-deprecated-declarations
            )
        endif()

        foreach(FLAG ${CMAKE_COMPILER_WARNINGS})
            if(NOT CMAKE_${COMPILER}_FLAGS MATCHES ${FLAG})
                set(CMAKE_${COMPILER}_FLAGS "${CMAKE_${COMPILER}_FLAGS} ${FLAG}")
            endif()
        endforeach()
    endforeach()
endif ()
