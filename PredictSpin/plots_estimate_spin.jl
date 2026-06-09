function plot_training_vs_test!(gl, plot_paulis, Y_test, Y_pred, Pm_dict, mse_per_pauli, σE)
    x_vals = range(-1, 1, length = 100)
    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        ax = Axis(gl[1, i],
            xlabel = "True value", ylabel = "Predicted value",
            title = "$(ps[1]) ⊗ $(ps[2]), MSE: $(round(mse_per_pauli[idx], digits=4)), σE: $(round(σE, digits=4))")
        scatter!(ax, Y_test[:, idx], Y_pred[:, idx],
            label = L"Y_{test} vs Y_{pred}", color = :orange, markersize = 15)
        lines!(ax, x_vals, x_vals, color = :black, label = "Optimal")
        axislegend(ax, position = :rb)
    end
end

function plot_training_vs_test(plot_paulis, Y_test, Y_pred, Pm_dict, mse_per_pauli, σE)
    fig = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))
    plot_training_vs_test!(fig.layout, plot_paulis, Y_test, Y_pred, Pm_dict, mse_per_pauli, σE)
    return fig
end

function plot_test_and_prediction!(gl, plot_paulis, Y_test, Y_pred, Pm_dict, mse_per_pauli)
    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_test[:, idx]))
        sort_idx = sortperm(Y_test[:, idx])
        ax = Axis(gl[1, i],
            ylabel = "Spin expectation value",
            title = "$(ps[1]) ⊗ $(ps[2]), MSE: $(round(mse_per_pauli[idx], digits=4))")
        scatter!(ax, x_range, Y_test[:, idx][sort_idx],
            label = L"Y_{test}", color = :orange, markersize = 15)
        scatter!(ax, x_range, Y_pred[:, idx][sort_idx],
            label = L"Y_{pred}", marker = :cross, color = :black, markersize = 12)
        axislegend(ax, position = :rb)
    end
end

function plot_test_and_prediction(plot_paulis, Y_test, Y_pred, Pm_dict, mse_per_pauli)
    fig = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))
    plot_test_and_prediction!(fig.layout, plot_paulis, Y_test, Y_pred, Pm_dict, mse_per_pauli)
    return fig
end

function plot_test!(gl, plot_paulis, Y_test, Pm_dict)
    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_test[:, idx]))
        sort_idx = sortperm(Y_test[:, idx])
        ax = Axis(gl[1, i],
            ylabel = "Spin expectation value",
            title = "$(ps[1]) ⊗ $(ps[2])")
        scatter!(ax, x_range, Y_test[:, idx][sort_idx],
            label = L"Y_{test}", color = :orange, markersize = 15)
        axislegend(ax, position = :rb)
    end
end

function plot_test(plot_paulis, Y_test, Pm_dict)
    fig = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))
    plot_test!(fig.layout, plot_paulis, Y_test, Pm_dict)
    return fig
end

function plot_training!(gl, plot_paulis, Y_train, Pm_dict)
    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_train[:, idx]))
        sort_idx = sortperm(Y_train[:, idx])
        ax = Axis(gl[1, i],
            ylabel = "Spin expectation value",
            title = "$(ps[1]) ⊗ $(ps[2])")
        scatter!(ax, x_range, Y_train[:, idx][sort_idx],
            label = L"Y_{train}", color = :orange, markersize = 15)
        axislegend(ax, position = :rb)
    end
end

function plot_training(plot_paulis, Y_train, Pm_dict)
    fig = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))
    plot_training!(fig.layout, plot_paulis, Y_train, Pm_dict)
    return fig
end

ps_labels = [("$(a) ⊗ $(b)")
             for a in ["σ0", "σx", "σy", "σz"], b in ["σ0", "σx", "σy", "σz"]]

function test_contains(S, ps)
    rank(Matrix(vcat(S, ps')), rtol = 1e-8) == rank(Matrix(S), rtol = 1e-8)
end

function test_S_row_space(S, Pm)
    for i in 1:16
        ps = Pm[:, i]
        print("$(ps_labels[i]) :")
        println(test_contains(S, ps) ? "True" : "False")
    end
end
