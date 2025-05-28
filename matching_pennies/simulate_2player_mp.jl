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

    dir = "/home/mehul/repos/MixtureAgentsModels/matching_pennies/data"

    player_1 = "JOA-M-0003"
    # Load the model from a pretrained model on real data
    model_1,_,agents_sim_1,agent_options_1,data_mp_1,ll_1 = loadfit("$dir/mp_fit_HMM_$player_1.mat")
    nstates_1 = 3    # number of hidden states

    # use pretrained model parameters
    β0_1 = model_1.β
    π0_1 = model_1.π
    A0_1 = model_1.A

    maxiter = 100  # maximum number of iterations for EM algorithm
    nstarts = 1    # number of reinitializations for EM algorithm
    tol = 1E-4     # tolerance for convergence of EM algorithm
    model_options_1 = ModelOptionsHMM(nstates=nstates_1,
                                      β0=β0_1,
                                      π0=π0_1,
                                      A0=A0_1,
                                      maxiter=maxiter,tol=tol,nstarts=nstarts)



    player_2 = "JOA-M-0014"

    # Load the model from a pretrained model on real data
    model_2,_,agents_sim_2,agent_options_2,data_mp_2,ll_2 = loadfit("$dir/mp_fit_HMM_$player_2.mat")
    nstates_2 = 3    # number of hidden states

    # use pretrained model parameters
    β0_2 = model_2.β
    π0_2 = model_2.π
    A0_2 = model_2.A

    maxiter = 100  # maximum number of iterations for EM algorithm
    nstarts = 1    # number of reinitializations for EM algorithm
    tol = 1E-4     # tolerance for convergence of EM algorithm
    model_options_2 = ModelOptionsHMM(nstates=nstates_2,
                                      β0=β0_2,
                                      π0=π0_2,
                                      A0=A0_2,
                                      maxiter=maxiter,tol=tol,nstarts=nstarts)


end


##- simulate data, fit model, and plot results -##
# simulate data, set init_model to false to use user-defined model parameters
data_sim_1,model_sim_1,agents_sim_1, data_sim_2,model_sim_2,agents_sim_2 = simulate_2player(sim_options,
                                                                                model_options_1, model_options_2,
                                                                                agent_options_1, agent_options_2)


                                     
# plot simulated model and hidden state probabilities in example sessions
# plot_model(model_sim,agents_sim,agent_options,data_sim)

# plot start sess 
# plot_model(model_sim,agents_sim,agent_options,data_sim;plot_example=true,sessions=1:1)

# plot middle sessions
plot_model(model_sim_1,agents_sim_1,agent_options_1,data_sim_1;plot_example=true)

# plot the last session
# plot_model(model_sim,agents_sim,agent_options,data_sim;plot_example=true,sessions=nsess:nsess)




