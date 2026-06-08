function plot_training_vs_test(plot_paulis, Y_test, Y_pred, Pm_dict, mean_squared_error, σE)
    x_vals = range(-1, 1, length = 100)
    fig = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))
    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        ax = Axis(fig[1, i],
            xlabel = "True value",
            ylabel = "Predicted value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2]), MSE: $(round(mean_squared_error[idx], digits=4)), σE: $(round(σE, digits=4))")
        scatter!(ax, Y_test[:, idx], Y_pred[:, idx],
            label = L"Y_{test} vs Y_{pred}", color = :orange, markersize = 15)
        lines!(ax, x_vals, x_vals, color = :black, label = "Optimal")
        axislegend(position = :rb)
    end

    display(fig)
end

function plot_training(plot_paulis, Y_train, Pm_dict)
    fig_train = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))

    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_train[:, idx]))
        ax = Axis(fig_train[1, i],
            ylabel = "Spin expectation value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2])")
        sort_idx = sortperm(Y_train[:, idx])
        scatter!(ax, x_range, Y_train[:, idx][sort_idx],
            label = L"Y_{train}", color = :orange, markersize = 15)
        axislegend(position = :rb)
    end
    display(fig_train)
end

function plot_test(plot_paulis, Y_test, Pm_dict)
    fig_test = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))

    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_test[:, idx]))
        ax = Axis(fig_test[1, i],
            ylabel = "Spin expectation value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2])")
        sort_idx = sortperm(Y_test[:, idx])
        scatter!(ax, x_range, Y_test[:, idx][sort_idx],
            label = L"Y_{test}", color = :orange, markersize = 15)
        axislegend(position = :rb)
    end
    display(fig_test)
end

function plot_test_and_prediction(plot_paulis, Y_test, Y_pred, Pm_dict, mean_squared_error)
    fig_test_pred = Figure(fontsize = 15, size = (200 + 300 * length(plot_paulis), 400))

    for (i, ps) in enumerate(plot_paulis)
        idx = Pm_dict[ps...]
        x_range = range(1, length(Y_test[:, idx]))
        ax = Axis(fig_test_pred[1, i],
            ylabel = "Spin expectation value",
            title = "Measurement: $(ps[1]) ⊗ $(ps[2]), MSE: $(round(mean_squared_error[idx], digits=4))")
        sort_idx = sortperm(Y_test[:, idx])
        scatter!(ax, x_range, Y_test[:, idx][sort_idx],
            label = L"Y_{test}", color = :orange, markersize = 15)
        scatter!(ax, x_range, Y_pred[:, idx][sort_idx],
            label = L"Y_{pred}", marker = :cross, color = :black, markersize = 12)
        axislegend(position = :rb)
    end
    display(fig_test_pred)
end

ps_labels = [("$(a) ⊗ $(b)")
             for a in ["σ0", "σx", "σy", "σz"], b in ["σ0", "σx", "σy", "σz"]]

function test_contains(S, ps)
    rank(Matrix(vcat(S, ps')), rtol = 1e-8) == rank(Matrix(S), rtol = 1e-8)
end

function test_S_row_space(S)
    for i in 1:16
        ps = Pm[:, i]
        print("$(ps_labels[i]) :")
        if test_contains(S, ps)
            println("True")
        else
            println("False")
        end
    end
end