/*
 * This file has been automatically generated by the jrl-cmakemodules.
 * Please see https://github.com/jrl-umi3218/jrl-cmakemodules/blob/master/warning.hh.cmake for details.
*/

#ifndef CONTACTBENCH_WARNING_HH
# define CONTACTBENCH_WARNING_HH

// Emits a warning in a portable way.
//
// To emit a warning, one can insert:
//
// #pragma message CONTACTBENCH_WARN("your warning message here")
//
// The use of this syntax is required as this is /not/ a standardized
// feature of C++ language or preprocessor, even if most of the
// compilers support it.

# define CONTACTBENCH_WARN_STRINGISE_IMPL(x) #x
# define CONTACTBENCH_WARN_STRINGISE(x) \
         CONTACTBENCH_WARN_STRINGISE_IMPL(x)
# ifdef __GNUC__
#   define CONTACTBENCH_WARN(exp) ("WARNING: " exp)
# else
#  ifdef _MSC_VER
#   define FILE_LINE_LINK __FILE__ "(" \
           CONTACTBENCH_WARN_STRINGISE(__LINE__) ") : "
#   define CONTACTBENCH_WARN(exp) (FILE_LINE_LINK "WARNING: " exp)
#  else
// If the compiler is not recognized, drop the feature.
#   define CONTACTBENCH_WARN(MSG) /* nothing */
#  endif // __MSVC__
# endif // __GNUC__

#endif //! CONTACTBENCH_WARNING_HH
