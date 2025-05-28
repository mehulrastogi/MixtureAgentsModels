"""
    GenericData{T1,T2,T3,T4} <: RatData

Example behavioral data struct. Replace `Generic` with task name when creating for a specific task. Required fields are ones that must be included for models to run correctly. Optional fields can be added for agent compatibility. Check an agents `next_Q!` function to see what behavior data it requires.
(Behavior data is not limited to specifically "rat" data, but can be from any subject. But aren't we all just differently sized rats at the end of the day?)
See `tasks/twostep_task.jl` for a comprehensive example of how to create a data struct for a specific task.

Per julia guidelines, it's recommended to explicitly define the type of each field in the constructor. This ensures faster compilation times and faster performance. 

# Required Fields:
- `ntrials`: total number of trials
- `nfree`: total number of `free` trials (e.g. the choice was not forced, as indicated by `forced`). defaults to `ntrials`
- `choices`: (1=primary, 2=secondary) choices on each trial. this is what is being predicted by the model
- `new_sess`: boolean vector that is `true` if trial marks the start of a session
- `new_sess_free`: boolean vector that marks the first free choice at the start of a session. defaults to `new_sess`
- `forced`: boolean vector that is `true` if the choice was forced. these trials are excluded from choice likelihood estimation. defaults to `falses(ntrials)`

# Generated Fields:
- `sess_inds`: vector of vectors of trial indices for each session
- `sess_inds_free`: vector of vectors of free trial indices for each session

# Example Optional Fields:
- `rewards`: (1=reward, -1=omission) rewards on each trial
- `stim`: some stimulus on each trial, e.g. sound frequency, click rate, etc. If the stimulus is side selective, use positive values for the primary side and negative values for the secondary side

"""
@with_kw struct GenericData{I<:Int,VI<:AbstractVector{Int},VB<:AbstractVector{Bool},VA<:AbstractVector} <: RatData 
    ntrials::I
    nfree::I = ntrials
    choices::VI
    rewards::VI
    # stim::VI
    new_sess::VB
    new_sess_free::VB = new_sess
    forced::VB = falses(ntrials)
    sess_inds::VA = get_sess_inds(new_sess)
    sess_inds_free::VA = get_sess_inds(new_sess_free)
    leftprobs::VB
    outcomes::VI
    
end

"""
    GenericData(data::D) where D <: Union{Dict,JSON3.Object}

Converts a dictionary or JSON object to a GenericData struct.
"""
function GenericData(data::D) where D <: Union{Dict,JSON3.Object}
    # remove "type" field if it exists; created when using `ratdata2dict`
    if "type" in keys(data)
        delete!(data,"type")
    end
    return GenericData(; data...)
end

"""
    GenericSim{T} <: SimOptions

Parameters for simulating task data. Required fields are used for generating sessions of fixed or variable length.
Add optional parameters specific to simulating task data

Required Fields:
- `nsess`: number of sessions
- `ntrials`: total number of trials 
- `mean_ntrials`: (overrides `ntrials`) mean number of trials/session if randomly drawing 
- `std_ntrials`: standard deviation of number of trials/session if randomly drawing. defaults to `Int(mean_ntrials/5)`
"""
@with_kw struct GenericSim{T} <: SimOptions where T
    nsess::T = 1
    ntrials::T = 1000
    mean_ntrials::T = 0
    std_ntrials::T = Int(mean_ntrials/5)
end

"""
    simulate_task(model, agents, sim_options, new_sess)

Task-specific function to simulate behavior data; to be called by `simulate_task(sim_options, model_options)` in `simulate_task.jl` 
Required for parameter recovery simulations

    Let's simulate the matching pennies task here (just a coin flip for now )
"""
function simulate_task(model::M,agents::Array{A},sim_options::GenericSim,new_sess::AbstractArray{Bool};seed=nothing, sim_agent_type="CoinFlip",learn_online=false) where {M <: MixtureAgentsModel, A <: Agent}
    @unpack ntrials = sim_options
    
    # code to simulate task data. 


    choices = Array{Int}(undef,ntrials) #choices by HMM
    rewards = Array{Int}(undef,ntrials)
    leftprobs = falses(ntrials) # boolean vector for left probabilities, if applicable
    outcomes = Array{Int}(undef,ntrials) # outcomes of the choices, e.g. 1=Left, 2=Right #choices by my agent

    data = GenericData(ntrials=ntrials, 
                        choices=choices, 
                        rewards=rewards, 
                        new_sess=new_sess, 
                        leftprobs=leftprobs,
                        outcomes=outcomes)

    
    if !isnothing(seed)
        Random.seed!(seed) # set seed for reproducibility
    end

    model_online = deepcopy(model) # make a copy of the model to use for online learning if specified
    agents_online = deepcopy(agents) # make a copy of the agents to use for online learning if specified

    x = zeros(length(agents),ntrials)
    Q = SizedMatrix{length(agents)}(init_Q(agents))
    z = zeros(Int,ntrials)
    pz = 1


    for t = 1:ntrials
        if new_sess[t]
            Q .= init_Q(agents_online) # reset Q values at the start of a new session
        end

        # set the xt to the difference of Q values for the 2  actions 
        x[:,t] = Q[:,1] .- Q[:,2]

        # determine the choice and outcome prob 
        pL,pz,z[t] = sim_choice_prob(model_online,x,t,pz,new_sess)

        # determine the choice by the agent 
        if rand() < pL
            choices[t] = 1 # primary choice (L)
        else
            choices[t] = 2 # secondary choice (R)
        end

        # determine the outcome 
        if sim_agent_type == "CoinFlip"
            # choose between 1 and 2 with equal probability
            if rand() < 0.5
                outcomes[t] = 1 # Left
            else
                outcomes[t] = 2 # Right
            end
        elseif sim_agent_type == "Alternate"
            # alternate between Left and Right
            if t % 2 == 0
                outcomes[t] = 1 # Left
            else
                outcomes[t] = 2 # Right
            end
        elseif sim_agent_type == "LeftBias"
            # choose Left with a probability of 0.7, Right with a probability of 0.3
            if rand() < 0.99
                outcomes[t] = 1 # Left
            else
                outcomes[t] = 2 # Right
            end
        elseif sim_agent_type == "MP"
            # check if the MatchingPenniesAgent is initialized if not, initialize it
            global mp_agent
            if !@isdefined(mp_agent)
                mp_agent = MatchingPenniesAgent()
            end

            if new_sess[t]
                initialize_patterns!(mp_agent) # initialize patterns at the start of a new session
            end

            computer_choice = choose_next_move(mp_agent)
            if computer_choice == "L"
                outcomes[t] = 1 # Left
            else
                outcomes[t] = 2 # Right
            end

        end

        # determine the reward based on the choice and outcome
        if choices[t] == outcomes[t]
            rewards[t] = -1 # omission if the choice matches the outcome
        else
            rewards[t] = 1 # reward if the choice does not match the outcome
        end

        if sim_agent_type == "MP"
            # need to update the MatchingPenniesAgent(computer) with the choice and reward
            choice_ = choices[t] == 1 ? "L" : "R"
            reward_ = rewards[t] == 1 ? 1 : 0
            update_agent!(mp_agent,choice_,reward_)
        end

        # update the Q values based on the choice and reward
        next_Q!(Q,agents_online,data,t)

        # check if the next trial is a new session
        next_new_session = false
        if t < ntrials
            next_new_session = new_sess[t+1]
        elseif t == ntrials
            next_new_session = true # last trial is always a new session
        end

        # we also need to update the entire model (the transitions and the agent weights)
        if learn_online && ( next_new_session)
            data_online = GenericData(ntrials=t, 
                                choices=choices[1:t], 
                                rewards=rewards[1:t], 
                                new_sess=new_sess[1:t], 
                                leftprobs=leftprobs[1:t],
                                outcomes=outcomes[1:t])
            # update the model with the new data

            # update the model with the new Q values and choices
            fit_symbs = [:α,:α]  # symbols for parameters to fit
            fit_params = [1,2,0]   # corresponding fit parameter index for each agent
            agent_options = AgentOptions(agents_online,fit_symbs,fit_params)

            nstates = 3   # number of hidden states
            maxiter = 300  # maximum number of iterations for EM algorithm (this probably won't converge, just for example. defaults to 300)
            nstarts = 5   # number of reinitializations for EM algorithm
            tol = 1E-4    # tolerance for convergence of EM algorithm
            model_options = ModelOptionsHMM(nstates=nstates,tol=tol,maxiter=maxiter,nstarts=nstarts)

            model_online,agents_online,ll = optimize(data_online,model_options,agent_options)
            @info "Trial $t: Log Likelihood = $ll"

            if true && next_new_session # plot after a session finishes
                _,Aplot,βplot,αplot = plot_model(model_online,agents_online,agent_options;return_plots=true)
                # get the latest session numbers 
                session_num = length(data_online.sess_inds_free)
                evaluation_sess_ids = data_online.sess_inds_free[session_num]
                @info "Plotting session $session_num : $evaluation_sess_ids "
                explot,_,exsessplot,reward_plot,_ = plot_gammas(model_online,agents_online,data_online;sessions=session_num:session_num)

                l = @layout [a{0.2w} b{0.6w} c{0.2w};d;e;f]

                session_plt = plot(Aplot,βplot,αplot,explot,exsessplot,reward_plot,layout=l,framestyle=:box, size=(2250,1875), margin=5mm,plot_title="Session $session_num")
                # put a title on the plot only the heading


                savefig(session_plt,"/home/mehul/repos/MixtureAgentsModels/matching_pennies/data/learn_sim/mp_fit_HMM_session_$session_num.png")
            end

            
        end

        

    end
    
    # return populated GenericData struct
    return data
end

"""
    simulate_2player_task(model_1,model_2,agents_1,agents_2,sim_options,new_sess)

Simulates a 2-player task where each player has their own model and agents.
This function is similar to `simulate_task`, but it simulates two players interacting with each other, where the outcome of one player's choice affects the other player's reward.
    PLayer 1: MisMatching Opponent (e.g. Matching Pennies)
    Player 2: Match Opponent (e.g. Mismatch Pennies)
"""
function simulate_2player_task(model_1::M,model_2::M,
                               agents_1::Array{A},agents_2::Array{A},
                               sim_options::GenericSim,
                               new_sess::AbstractArray{Bool};
                               seed=nothing) where {M <: MixtureAgentsModel, A <: Agent}
    @unpack ntrials = sim_options
    
    # code to simulate task data. 


    choices_1 = Array{Int}(undef,ntrials) #choices by HMM
    rewards_1 = Array{Int}(undef,ntrials)
    leftprobs_1 = falses(ntrials) # boolean vector for left probabilities, if applicable
    outcomes_1 = Array{Int}(undef,ntrials) # outcomes of the choices, e.g. 1=Left, 2=Right #choices by my agent

    data_1 = GenericData(ntrials=ntrials, 
                        choices=choices_1,
                        rewards=rewards_1,
                        new_sess=new_sess,
                        leftprobs=leftprobs_1,
                        outcomes=outcomes_1)

    choices_2 = Array{Int}(undef,ntrials) #choices by HMM
    rewards_2 = Array{Int}(undef,ntrials)
    leftprobs_2 = falses(ntrials) # boolean vector for left probabilities, if applicable
    outcomes_2 = Array{Int}(undef,ntrials) # outcomes of the choices, e.g. 1=Left, 2=Right #choices by my agent

    data_2 = GenericData(ntrials=ntrials, 
                        choices=choices_2,
                        rewards=rewards_2,
                        new_sess=new_sess,
                        leftprobs=leftprobs_2,
                        outcomes=outcomes_2)

    
    if !isnothing(seed)
        Random.seed!(seed) # set seed for reproducibility
    end

    x_1 = zeros(length(agents_1),ntrials)
    Q_1 = SizedMatrix{length(agents_1)}(init_Q(agents_1))
    z_1 = zeros(Int,ntrials)
    pz_1 = 1

    x_2 = zeros(length(agents_2),ntrials)
    Q_2 = SizedMatrix{length(agents_2)}(init_Q(agents_2))
    z_2 = zeros(Int,ntrials)
    pz_2 = 1




    for t = 1:ntrials
        if new_sess[t]
            Q_1 .= init_Q(agents_1) # reset Q values at the start of a new session
            Q_2 .= init_Q(agents_2) # reset Q values at the start of a new session
        end

        # set the xt to the difference of Q values for the 2  actions 
        x_1[:,t] = Q_1[:,1] .- Q_1[:,2]
        x_2[:,t] = Q_2[:,1] .- Q_2[:,2]

        # determine the choice and outcome prob 
        pL_1,pz_1,z_1[t] = sim_choice_prob(model_1,x_1,t,pz_1,new_sess)
        pL_2,pz_2,z_2[t] = sim_choice_prob(model_2,x_2,t,pz_2,new_sess)

        # determine the choice by the agents_1
        if rand() < pL_1
            choices_1[t] = 1 # primary choice (L)
        else
            choices_1[t] = 2 # secondary choice (R)
        end

        # determine the choice by the agents_2
        if rand() < pL_2
            choices_2[t] = 1 # primary choice (L)
        else
            choices_2[t] = 2 # secondary choice (R)
        end

        # set the outcomes based on the choices of both agents
        outcomes_1[t] = choices_2[t] # agent 1's outcome is based on agent 2's choice
        outcomes_2[t] = choices_1[t] # agent 2's outcome is based on agent 1's choice

        # determine the reward based on the choice and outcome
        if choices_1[t] == outcomes_1[t]
            rewards_1[t] = -1 # omission if the choice matches the outcome
        else
            rewards_1[t] = 1 # reward if the choice does not match the outcome
        end

        if choices_2[t] != outcomes_2[t]
            rewards_2[t] = -1 # omission if the choice matches the outcome
        else
            rewards_2[t] = 1 # reward if the choice does not match the outcome
        end
        

        # determine the reward based on the choice and outcome
        if choices[t] == outcomes[t]
            rewards[t] = -1 # omission if the choice matches the outcome
        else
            rewards[t] = 1 # reward if the choice does not match the outcome
        end


        next_Q!(Q_1,agents_1,data_1,t)
        next_Q!(Q_2,agents_2,data_2,t)
        
    end
    
    # return populated GenericData struct
    return data_1, data_2
end



"""
    load_EXAMPLE(file::String,...)

Custom function to load behaiovral data for EXAMPLE task from file. 

If it is a .mat file, you can use `matread` to load the data, and then transform it into variable types required by GenericData struct. 
You can also use `CSV.read` to load a .csv file into a DataFrame, and then convert it to a dictionary for GenericData struct.
See "data/example_task_data.csv" for an example of how to format a .csv file for GenericData struct, and "data/example_task_data.mat" for an example of how to format a .mat file for GenericData struct.
"""
function load_generic(file::String="data/example_task_data.csv")
    ext = split(file,'.')[end] # get file extension

    if ext == "mat"
        return load_generic_mat(file)

    elseif ext == "csv"
        return load_generic_csv(file)
    end

end

function load_generic_mat(file="data/example_task_data.mat")
        # code to load data from file
        matdata = matread(file)
        varname = collect(keys(matdata))[1] # this assumes there's only one variable in the .mat file
        matdata = matdata[varname]
        # package results into dictionary D for easy conversion to GenericData
        D = Dict{Symbol,Any}(
            :ntrials => length(matdata["choices"]), # number of trials
            :choices => vec(Int.(matdata["choices"])), # convert to int, make sure it's a vector
            :rewards => vec(Int.(matdata["rewards"])), # convert to int, make sure it's a vector
            # :stim => vec(Int.(matdata["stim"])), # convert to int, make sure it's a vector
            :new_sess => vec(matdata["new_sess"] .== 1)), # convert to boolean, make sure it's a vector
            :leftprobs => vec(matdata["leftprobs"] .== 1) # convert to boolean, make sure it's a vector
    
        # return GenericData struct
        return GenericData(D)
end

function load_generic_csv(file="data/example_task_data.csv")
    # load into dataframe
    df = CSV.read(file,DataFrame)
    # convert to dict
    # you may need to explicitly re-type things to avoid weirdly loaded types, like "String7" instead of "String". see "pclicks_task.jl" for an example
    # D = Dict{Symbol,Any}(pairs(eachcol(df)))
    D = Dict{Symbol,Any}(
        :choices=>Int.(df.choices),
        :rewards=>Int.(df.rewards),
        # :stim=>Int.(df.stim),
        :new_sess=>Bool.(df.new_sess),
        :leftprobs => Bool.(df.leftprobs),
        :outcomes => Int.(df.outcomes) # outcomes are the  choices computer has made 
    )

    D[:ntrials] = length(D[:choices])
    # D[:new_sess] = D[:new_sess] .== 1 # convert to boolean vector

    return GenericData(D)
end


