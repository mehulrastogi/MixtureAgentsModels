using MAT, MixtureAgentsModels

file = "/home/mehul/repos/MixtureAgentsModels/data/MBB2017_behavioral_dataset.mat"
rat = 17              # rat number
data = load_twostep(file,rat)

matdata = matread(file)