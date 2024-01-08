#ifndef COMPILE_TIME_ARRAY_HEADER
#define COMPILE_TIME_ARRAY_HEADER

#include <array>

namespace CompileTimeArray
{
    namespace
    {
        template <typename T, std::size_t ... Is>
        constexpr std::array<T, sizeof...(Is)> create_array(T value, std::index_sequence<Is...>)
        {
            // cast Is to void to remove the warning: unused value
            return { {(static_cast<void>(Is), value)...} };
        }
        template <typename T, typename... Args, std::size_t ... Is>
        constexpr std::array<T, sizeof...(Is)> create_array_from_args(Args... values, std::index_sequence<Is...>)
        {
            // cast Is to void to remove the warning: unused value
            return { {(static_cast<void>(Is), T(values...))...} };
        }
    }

    template <typename T, std::size_t N>
    constexpr std::array<T, N> uniform_array(const T& value)
    {
        return create_array(value, std::make_index_sequence<N>());
    }

    template <typename T, std::size_t N, typename... Args>
    constexpr std::array<T, N> uniform_array_from_args(Args... values)
    {
        return create_array_from_args<T, Args...>(values..., std::make_index_sequence<N>());
    }
}

#endif