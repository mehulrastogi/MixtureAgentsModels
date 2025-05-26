using MixtureAgentsModels # import package


########################################################################################
######################              Fit to real data            ########################
########################################################################################
##- load in data -##
# load two-step data using custom function that parses twostep .mat files
file = "/home/mehul/repos/MixtureAgentsModels/matching_pennies/data/matchingpennies_formatted.csv" # path to data file           # rat number
data_mp = load_generic(file)

##- define agent options -##
# agent vector
agents = [MFrewardB(), MFchoiceB(),Bias()]
fit_symbs = [:α,:α]  # symbols for parameters to fit
fit_params = [1,2,0]   # corresponding fit parameter index for each agent
agent_options = AgentOptions(agents,fit_symbs,fit_params)

##- define HMM options -##
nstates = 3   # number of hidden states
maxiter = 1000  # maximum number of iterations for EM algorithm (this probably won't converge, just for example. defaults to 300)
nstarts = 5   # number of reinitializations for EM algorithm
tol = 1E-4    # tolerance for convergence of EM algorithm
model_options = ModelOptionsHMM(nstates=nstates,tol=tol,maxiter=maxiter,nstarts=nstarts)


# # load fit from .mat file
# model,model_options,agents,agent_options,data_mp,ll = loadfit("/home/mehul/repos/MixtureAgentsModels/matching_pennies/data/mp_fit_HMM.mat")


##- fit model to data -##
model,agents,ll = optimize(data_mp,model_options,agent_options)
# need more iterations? restart from previous fit
model,agents,ll = optimize(data_mp,model,agents,model_options,agent_options)

# plot_model(model,agents,agent_options)

# plot fit model and hidden state probabilities in example sessions
plot_model(model,agents,agent_options,data_mp)
# save fit to .mat file
savefit("/home/mehul/repos/MixtureAgentsModels/matching_pennies/data/mp_fit_HMM.mat",model,model_options,agents,agent_options,data_mp)
