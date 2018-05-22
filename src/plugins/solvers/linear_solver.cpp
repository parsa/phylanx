// Copyright (c) 2018 Shahrzad Shirzad
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/plugins/solvers/linear_solver.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>

#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <blaze/Math.h>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace execution_tree { namespace primitives
{
///////////////////////////////////////////////////////////////////////////////
#define PHYLANX_LIN_MATCH_DATA(name)                                           \
    hpx::util::make_tuple(name, std::vector<std::string>{name "(_1, _2)"},     \
        &create_linear_solver, &create_primitive<linear_solver>)
    /**/

    std::vector<match_pattern_type> const linear_solver::match_data = {
        PHYLANX_LIN_MATCH_DATA("linear_solver_lu"),
        PHYLANX_LIN_MATCH_DATA("linear_solver_ldlt_u"),
        PHYLANX_LIN_MATCH_DATA("linear_solver_ldlt_l")};

#undef PHYLANX_LIN_MATCH_DATA

    ///////////////////////////////////////////////////////////////////////////
    linear_solver::vector_function_ptr linear_solver::get_lin_solver_map(
        std::string const& name) const
    {
        static std::map<std::string, vector_function_ptr> lin_solver = {
            {"linear_solver_lu",
                [](args_type&& args) -> arg_type {
                    storage2d_type A{blaze::trans(args[0].matrix())};
                    storage1d_type b{args[1].vector()};
                    const std::unique_ptr<int[]> ipiv(new int[b.size()]);
                    blaze::gesv(A, b, ipiv.get());
                    return arg_type{std::move(b)};
                }},
            {"linear_solver_ldlt_u",
                [](args_type&& args) -> arg_type {
                    storage2d_type A{blaze::trans(args[0].matrix())};
                    storage1d_type b{args[1].vector()};
                    const std::unique_ptr<int[]> ipiv(new int[b.size()]);
                    blaze::sysv(A, b, 'U', ipiv.get());
                    return arg_type{std::move(b)};
                }},
            {"linear_solver_ldlt_l", [](args_type&& args) -> arg_type {
                 storage2d_type A{blaze::trans(args[0].matrix())};
                 storage1d_type b{args[1].vector()};
                 const std::unique_ptr<int[]> ipiv(new int[b.size()]);
                 blaze::sysv(A, b, 'L', ipiv.get());
                 return arg_type{std::move(b)};
             }}};
        return lin_solver[name];
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        std::string extract_function_name(std::string const& name)
        {
            std::string::size_type p = name.find_first_of("$");
            if (p != std::string::npos)
            {
                return name.substr(0, p);
            }
            return name;
        }
    }

    linear_solver::linear_solver(
        std::vector<primitive_argument_type> && operands,
        std::string const& name, std::string const& codename)
      : primitive_component_base(std::move(operands), name, codename)
    {
        std::string func_name = detail::extract_function_name(name);

        func_ = get_lin_solver_map(func_name);

        HPX_ASSERT(func_ != nullptr);
    }

    ///////////////////////////////////////////////////////////////////////////
    primitive_argument_type linear_solver::calculate_linear_solver(
        args_type && op) const
    {
        return primitive_argument_type{func_(std::move(op))};
    }

    hpx::future<primitive_argument_type> linear_solver::eval(
        std::vector<primitive_argument_type> const& operands,
        std::vector<primitive_argument_type> const& args) const
    {
        if (operands.size() != 2)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "linear_solver::eval",
                execution_tree::generate_error_message(
                    "the linear_solver  primitive "
                    "requires exactly two operands ",
                    name_, codename_));
        }

        if (!valid(operands[0]) || !valid(operands[1]))
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "linear_solver_operation::eval",
                execution_tree::generate_error_message(
                    "the linear_solver primitive requires "
                    "that the arguments given by the operands "
                    "array are valid",
                    name_, codename_));
        }

        auto this_ = this->shared_from_this();
        return hpx::dataflow(hpx::launch::sync,
            hpx::util::unwrapping(
                [this_](args_type&& args) -> primitive_argument_type {
                    if (args[0].num_dimensions() != 2 ||
                        args[1].num_dimensions() != 1)
                    {
                        HPX_THROW_EXCEPTION(hpx::bad_parameter,
                            "linear_solver_operation::eval",
                            execution_tree::generate_error_message(
                                "the linear_solver_operation primitive "
                                "requires "
                                "that first operand to be a mtarix and "
                                "the second "
                                "operand to be a vector",
                                this_->name_, this_->codename_));
                    }

                    return this_->calculate_linear_solver(std::move(args));
                }),
            detail::map_operands(operands, functional::numeric_operand{}, args,
                this_->name_, this_->codename_));
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<primitive_argument_type> linear_solver::eval(
        std::vector<primitive_argument_type> const& args) const
    {
        if (operands_.empty())
        {
            return eval(args, noargs);
        }
        return eval(operands_, args);
    }
}}}
