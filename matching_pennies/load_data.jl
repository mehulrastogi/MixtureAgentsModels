# load a df using a parquet file 

using DataFrames, DataFramesMeta, Parquet2, CSV, Statistics, CairoMakie

# load the data from '/mnt/ogma/delab/matchingpennies/matchingpennies_datatable.parquet'

ds = Parquet2.Dataset("/home/mehul/repos/MixtureAgentsModels/matching_pennies/data/matchingpennies_datatable.parquet")

df = DataFrame(ds)

# get a subset of the data subjid in ['JOA-M-0002','JOA-M-0003', ]

df = @chain df begin
    @transform(
        :sessid = Int16.(:sessid),
        :trialnum = Int16.(:trialnum),
        :subjid = String.(:subjid),

    )
    @rsubset(:subjid in ["JOA-M-0003"])
    @rsubset(:protocol == "MatchingPennies")
    @rsubset(:choice in ["L","R"]) # 0 = left, 1 = right
    @select(:subjid,:protocol,:sessid, :trialnum, :choice,:reward,:comp_prediction,:RT,:leftlicknum,:rightlicknum,:p_stochastic, :p_leftbias, :p_rightbias)
end


num_unique_sessions = length(unique(df.sessid))

# get the reward rate for each session and each animal 
reward_rate = @chain df begin
    @select(:subjid, :sessid,:reward)
    @groupby(:subjid, :sessid)
    @combine(:reward_rate = mean(:reward))
    @orderby(:subjid, :sessid)
end


# # plot the reward rate for each session and each animal using CairoMakie line plot
# CairoMakie.activate!(type = "svg")
# fig = Figure()

# ax = Axis(fig[1, 1], title = "Reward Rate", xlabel = "Session", ylabel = "Reward Rate", xticks = 1:num_unique_sessions, yticks = 0:0.1:1)
# lines!(ax, reward_rate.sessid, reward_rate.reward_rate, color = :blue, linewidth = 2)
# CairoMakie.scatter!(ax, reward_rate.sessid, reward_rate.reward_rate, color = :red)

# # save the figure
# # show the figure
# display(fig)

sess_id_current = -1

function detect_change(sess_id_row)
    global sess_id_current

    if sess_id_row != sess_id_current
        sess_id_current = sess_id_row
        return true
    else
        return false
    end
    
end

# writing the data in a particular format 

df_formatted = @chain df begin
   
    @rtransform(
        :choices = ifelse(:choice == "R", 1,2), # convert choice to 1 for left, 2 for right
        :rewards = ifelse(:reward == 1, 1,-1), # convert reward to 1 for reward, -1 for no reward
        :new_sess = detect_change(:sessid), # detect change in session id
        :leftprobs = 1
    )
    @select(
        :choices, :rewards, :new_sess, :leftprobs
    )
end

# save this parquet file
using CSV 

CSV.write("/home/mehul/repos/MixtureAgentsModels/matching_pennies/data/matchingpennies_formatted.csv", df_formatted)

