using MixtureAgentsModels # import package

########################################################################################
####################              Fit to simulated data            #####################
########################################################################################
##- define simulation options -##
nsess = 20          # number of sessions
ntrials = 100      # total number of trials
mean_ntrials = 450  # mean number of trials per session; will override ntrials
sim_options = GenericSim(nsess=nsess,ntrials=ntrials,mean_ntrials=mean_ntrials)


# ================================
# ------ EXAMPLE DEFINIATION OF OPTIONS------------
# include the following lines to define the options for the agents and HMM model
##- define agent options -##
# # agent vector, user set learning rates for RL agents
# agents_sim = [MFrewardB(α=0.40),MFchoiceB(α=0.50),Bias()]
# fit_symbs = [:α,:α]  # symbols for each parameter to fit
# fit_params = [1,2,0]   # index linking fit_symbs to corresponding agent in agents_sim
# agent_options = AgentOptions(agents_sim,fit_symbs,fit_params)


# ##- define HMM options -##
# nstates = 2    # number of hidden states
# # user set initial values for HMM parameters to be used for simulation
# β0 = [0.79 1.25; -0.82 -0.19; -0.01 0.1] # (nagents x nstates) matrix of agent weights
# π0 = [1.0,0.0] # (nstates x 1) vector of initial state probabilities
# A0 = [0.9 0.1; 0.01 0.99] # (nstates x nstates) transition SizedMatrix
#  -------------------------------------------
# ================================

load_pretrained = true
if load_pretrained

    # agents_sim = [MFrewardB(α=0.40),MFchoiceB(α=0.50),Bias()]
    # fit_symbs = [:α,:α]  # symbols for each parameter to fit
    # fit_params = [1,2,0]   # index linking fit_symbs to corresponding agent in agents_sim
    # agent_options = AgentOptions(agents_sim,fit_symbs,fit_params)

    # Load the model from a pretrained model on real data
    model,_,agents_sim,agent_options,data_mp,ll = loadfit("/home/mehul/repos/MixtureAgentsModels/matching_pennies/data/mp_fit_HMM.mat")
    
    nstates = 3    # number of hidden states
    # use pretrained model parameters
    β0 = model.β
    π0 = model.π
    A0 = model.A
end

# left biased agent
if false

    # purely biaased agent
    ##- define agent options -##
    # agent vector, user set learning rates for RL agents
    agents_sim = [MFrewardB(α=0.40),MFchoiceB(α=0.50),Bias()]
    fit_symbs = [:α,:α]  # symbols for each parameter to fit
    fit_params = [1,0]   # index linking fit_symbs to corresponding agent in agents_sim
    agent_options = AgentOptions(agents_sim,fit_symbs,fit_params)


    ##- define HMM options -##
    nstates = 2    # number of hidden states
    # # user set initial values for HMM parameters to be used for simulation
    β0 = [0.0 0.0; 0.0 0.0; 10.0 10.0] # (nagents x nstates) matrix of agent weights
    π0 = [1.0,0.0] # (nstates x 1) vector of initial state probabilities
    A0 = [1.0 0.0; 0 1.0] # (nstates x nstates) transition SizedMatrix

end

# alternate agent
if false

    # purely alternate agent that alternates between left and right
    ##- define agent options -##
    # agent vector, user set learning rates for RL agents
    agents_sim = [MFrewardB(α=0.40),MFchoiceB(α=0.50),Bias()]
    fit_symbs = [:α,:α]  # symbols for each parameter to fit
    fit_params = [1,0]   # index linking fit_symbs to corresponding agent in agents_sim
    agent_options = AgentOptions(agents_sim,fit_symbs,fit_params)

    ##- define HMM options -##
    nstates = 2    # number of hidden states
    # user set initial values for HMM parameters to be used for simulation
    β0 = [0.0 0.0; 0.0 0.0; 10.0 -10.0] # (nagents x nstates) matrix of agent weights
    π0 = [1.0,0.0] # (nstates x 1) vector of initial state probabilities (start left )
    A0 = [0.0 1.0; 1.0 0.0] # (nstates x nstates) transition SizedMatrix
end




maxiter = 100  # maximum number of iterations for EM algorithm
nstarts = 1    # number of reinitializations for EM algorithm
tol = 1E-4     # tolerance for convergence of EM algorithm
model_options = ModelOptionsHMM(nstates=nstates,β0=β0,π0=π0,A0=A0,maxiter=maxiter,tol=tol,nstarts=nstarts)

##- simulate data, fit model, and plot results -##
# simulate data, set init_model to false to use user-defined model parameters
data_sim,model_sim,agents_sim = simulate(sim_options,
                                     model_options,
                                     agent_options;
                                     task="Generic",
                                     init_model=false,
                                     sim_agent_type="CoinFlip",
                                     learn_online=true)
# plot simulated model and hidden state probabilities in example sessions
# plot_model(model_sim,agents_sim,agent_options,data_sim)

# plot start sess 
# plot_model(model_sim,agents_sim,agent_options,data_sim;plot_example=true,sessions=1:1)

# plot middle sessions
# plot_model(model_sim,agents_sim,agent_options,data_sim;plot_example=true)

# plot the last session
# plot_model(model_sim,agents_sim,agent_options,data_sim;plot_example=true,sessions=nsess:nsess)


# plot_β(model_sim,agents_sim)
# plot_tr(model_sim,agents_sim)
# plot_A(model_sim)
# need to plot the choices !!


# fit_simulated_data = false

# if fit_simulated_data
#     # Fitting the model to the simulated data
#     agents = [MFrewardB(), MFchoiceB(),Bias()]
#     fit_symbs = [:α,:α]  # symbols for parameters to fit
#     fit_params = [1,2,0]   # corresponding fit parameter index for each agent
#     agent_options = AgentOptions(agents,fit_symbs,fit_params)

#     ##- define HMM options -##
#     nstates = 3   # number of hidden states
#     maxiter = 1000  # maximum number of iterations for EM algorithm (this probably won't converge, just for example. defaults to 300)
#     nstarts = 5   # number of reinitializations for EM algorithm
#     tol = 1E-4    # tolerance for convergence of EM algorithm
#     model_options = ModelOptionsHMM(nstates=nstates,tol=tol,maxiter=maxiter,nstarts=nstarts)


#     # # # fit model to simulated data
#     model_fit,agents_fit,ll_fit = optimize(data_sim,model_options,agent_options;disp_iter=10) # only print every 10th iteration
#     # # # plot fit model and hidden state probabilities in example sessions
#     plot_model(model_fit,agents_fit,agent_options,data_sim)
# end

