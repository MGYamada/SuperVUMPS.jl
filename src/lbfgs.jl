inner(x, v) = real(dot(x, v))

function riemannian_lbfgs(f::Function, grad!::Function, x0;
    maxiter::Int = 200,
    m::Int = 10,
    tol::Float64 = 1e-6,
    α0::Float64 = 1.0,
    verbose::Bool = false)
    x = copy(x0)
    x ./= norm(x)

    g = similar(x)
    grad!(g, x)
    g .= project_tangent(x, g)

    fx = f(x)

    s_list = Vector{typeof(x0)}()
    y_list = Vector{typeof(x0)}()
    ρ_list = Float64[]

    verbose && @printf("Iter %4d: f = %.6e, ‖grad‖ = %.3e\n", 0, fx, norm(g))

    iters = 0
    for k in 1:maxiter
        iters = k

        if norm(g) < tol
            break
        end

        # ========== 2-loop recursion ==========
        q = copy(g)
        ℓ = length(s_list)
        α = zeros(Float64, ℓ)
        for i in ℓ:-1:1
            α[i] = ρ_list[i] * inner(s_list[i], q)
            q .-= α[i] .* y_list[i]
        end
        γ = 1.0
        if ℓ > 0
            s_last = s_list[end]
            y_last = y_list[end]
            γ = inner(s_last, y_last) / inner(y_last, y_last)
        end
        r = γ .* q
        for i in 1:ℓ
            β = ρ_list[i] * inner(y_list[i], r)
            r .+= (α[i] - β) .* s_list[i]
        end
        p = -project_tangent(x, r)
        if norm(p) < 1e-12
            verbose && println("Search direction too small.")
            break
        end

        # ========== Armijo backtracking line search ==========
        αk = α0
        c = 1e-4
        max_bt = 20
        x_new = retract(x, αk * p)
        fx_new = f(x_new)
        for bt in 1:max_bt
            x_try = retract(x, αk * p)
            fx_try = f(x_try)
            if fx_try <= fx + c * αk * inner(g, p)
                x_new = x_try
                fx_new = fx_try
                break
            end
            αk *= 0.5
        end

        g_new = similar(g)
        grad!(g_new, x_new)
        g_new .= project_tangent(x_new, g_new)

        s = transport(x, x_new, αk * p)
        g_old_trans = transport(x, x_new, g)
        y = g_new .- g_old_trans

        sy = inner(s, y)
        push!(s_list, copy(s))
        push!(y_list, copy(y))
        push!(ρ_list, 1.0 / sy)

        if length(s_list) > m
            popfirst!(s_list); popfirst!(y_list); popfirst!(ρ_list)
        end

        x .= x_new
        g .= g_new
        fx = fx_new

        verbose && @printf("Iter %4d: f = %.6e, ‖grad‖ = %.3e, α = %.2e\n",
                           k, fx, norm(g), αk)
    end

    x, fx, iters
end