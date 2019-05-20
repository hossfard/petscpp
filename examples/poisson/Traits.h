#ifndef TRAITS_H_
#define TRAITS_H_

#include <array>

#define GENERATE_HAS_MEMBER(member)                                     \
  template<typename T>                                                  \
  struct has_member_##member                                            \
  {                                                                     \
  private:                                                              \
    using yes = std::true_type;                                         \
    using no  = std::false_type;                                        \
                                                                        \
    template<typename U> static auto test(int)                          \
      -> decltype(std::declval<U>().one == 1, yes());                   \
                                                                        \
    template<typename> static no test(...);                             \
                                                                        \
  public:                                                               \
    static constexpr bool value = std::is_same<decltype(test<T>(0)),yes>::value; \
  };                                                                    \


namespace openfem{

  template <typename>
  struct array_size;

  template <typename T, size_t N>
  struct array_size<std::array<T,N>>{
    static size_t const value = N;
  };

}


#endif /* TRAITS_H_ */


