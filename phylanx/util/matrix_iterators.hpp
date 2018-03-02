// Copyright (c) 2018 Parsa Amini
// Copyright (c) 2018 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PHYLANX_MATRIX_ITERATORS)
#define PHYLANX_MATRIX_ITERATORS

#include <algorithm>
#include <cstddef>
#include <utility>

#include <blaze/Math.h>

namespace blaze
{
    // BADBAD: This overload of swap is necessary to work around the problems
    //         caused by matrix_row_iterator not being a real random access
    //         iterator. Dereferencing matrix_row_iterator does not yield a
    //         true reference but only a temporary blaze::Row holding true
    //         references.
    //
    // A real fix for this problem is proposed in PR0022R0
    // (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0022r0.html)
    //
    template <typename T>
    HPX_FORCEINLINE
        void swap(Row<T>&& x, Row<T>&& y)
    {
        for (auto a = x.begin(), b = y.begin(); a != x.end() && b != y.end();
             ++a, ++b)
        {
            using std::iter_swap;

            iter_swap(a, b);
        }
    }
}

namespace phylanx { namespace util
{

    template <typename T>
    class matrix_row_iterator
        : public hpx::util::iterator_facade<
            matrix_row_iterator<T>,
            blaze::Row<T>,
            std::random_access_iterator_tag,
            blaze::Row<T>>
    {
    public:
        explicit matrix_row_iterator(T& t, const std::size_t index = 0)
            : data_(&t)
            , index_(index)
        {
        }

    private:
        friend class hpx::util::iterator_core_access;

        void increment()
        {
            ++index_;
        }

        void decrement()
        {
            --index_;
        }

        void advance(std::size_t n)
        {
            index_ += n;
        }

        bool equal(matrix_row_iterator const& other) const
        {
            return index_ == other.index_;
        }

        blaze::Row<T> dereference() const
        {
            return blaze::row(*data_, index_);
        }

        std::ptrdiff_t distance_to(matrix_row_iterator const& other) const
        {
            return other.index_ - index_;
        }

    private:
        T* data_;
        std::size_t index_;
    };

    template <typename T>
    class matrix_column_iterator
        : public hpx::util::iterator_facade<
        matrix_column_iterator<T>,
        blaze::Column<T>,
        std::random_access_iterator_tag,
        blaze::Column<T>>
    {
    public:
        explicit matrix_column_iterator(T& t, const std::size_t index = 0)
            : data_(&t)
            , index_(index)
        {
        }

    private:
        friend class hpx::util::iterator_core_access;

        void increment()
        {
            ++index_;
        }

        void decrement()
        {
            --index_;
        }

        void advance(std::size_t n)
        {
            index_ += n;
        }

        bool equal(matrix_column_iterator const& other) const
        {
            return index_ == other.index_;
        }

        blaze::Column<T> dereference() const
        {
            return blaze::column(*data_, index_);
        }

        std::ptrdiff_t distance_to(matrix_column_iterator const& other) const
        {
            return other.index_ - index_;
        }

    private:
        T* data_;
        std::size_t index_;
    };
}}

#endif
