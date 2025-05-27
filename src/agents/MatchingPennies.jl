using Distributions, HypothesisTests, StatsAPI

mutable struct MatchingPenniesAgent
    # Choice-only pattern tracking
    choice_patterns::Dict{String, Tuple{Int, Int, Float64}}
    # Choice-reward pattern tracking
    cr_patterns::Dict{String, Tuple{Int, Int, Float64}}
    # History
    choice_history::Vector{String}
    reward_history::Vector{Int}
    # Configuration
    alpha::Float64
    trials_back::Int
    initialized::Bool
    best_pvalue::Float64
    best_bias::Float64
    
    function MatchingPenniesAgent(alpha=0.05, trials_back=5)
        new(Dict{String, Tuple{Int, Int, Float64}}(), 
            Dict{String, Tuple{Int, Int, Float64}}(), 
            String[], Int[], alpha, trials_back, false, 1.0, 0.5)
    end
end

function initialize_patterns!(agent::MatchingPenniesAgent)
    # Initialize choice-only patterns
    # For 0-back (global stats)
    agent.choice_patterns[""] = (0, 0, 1.0)
    
    # For 1-5 back
    for n in 1:agent.trials_back
        # Generate all possible patterns of length n
        for pattern in generate_choice_patterns(n)
            agent.choice_patterns[pattern] = (0, 0, 1.0)
        end
    end
    
    # Initialize choice-reward patterns
    for n in 1:agent.trials_back
        # Generate all possible patterns of length n
        for pattern in generate_cr_patterns(n)
            agent.cr_patterns[pattern] = (0, 0, 1.0)
        end
    end
    
    agent.initialized = true
end

function generate_choice_patterns(n)
    choices = ["L", "R"]
    if n == 1
        return choices
    else
        patterns = String[]
        for prev_pattern in generate_choice_patterns(n-1)
            for choice in choices
                push!(patterns, string(prev_pattern, " ", choice))
            end
        end
        return patterns
    end
end

function generate_cr_patterns(n)
    cr_pairs = ["L 0", "L 1", "R 0", "R 1"]
    if n == 1
        return cr_pairs
    else
        patterns = String[]
        for prev_pattern in generate_cr_patterns(n-1)
            for pair in cr_pairs
                push!(patterns, string(prev_pattern, " ", pair))
            end
        end
        return patterns
    end
end

function update_agent!(agent::MatchingPenniesAgent, choice::String, reward::Int)
    # Initialize patterns if first call
    if !agent.initialized
        initialize_patterns!(agent)
    end
    
    # Only update patterns if this is a valid choice (not a miss)
    if choice in ["L", "R"]
        # Update choice-only patterns
        # First, update global stats
        l_count, total, _ = agent.choice_patterns[""]
        new_l_count = l_count + (choice == "L" ? 1 : 0)
        new_total = total + 1
        p_value = total > 0 ? pvalue(BinomialTest(new_l_count, new_total, 0.5)) : 1.0
        agent.choice_patterns[""] = (new_l_count, new_total, p_value)
        
        # Update pattern-specific stats
        len = length(agent.choice_history)
        for n in 1:min(agent.trials_back, len)
            # Get context (n previous choices)
            if len >= n
                context = join(agent.choice_history[end-(n-1):end], " ")
                
                # Update stats
                if haskey(agent.choice_patterns, context)
                    l_count, total, _ = agent.choice_patterns[context]
                    new_l_count = l_count + (choice == "L" ? 1 : 0)
                    new_total = total + 1
                    p_value = pvalue(BinomialTest(new_l_count, new_total, 0.5))
                    agent.choice_patterns[context] = (new_l_count, new_total, p_value)
                end
            end
        end
        
        # Update choice-reward patterns
        len = length(agent.choice_history)
        for n in 1:min(agent.trials_back, len)
            # Get context (n previous choice-reward pairs)
            if len >= n
                context_parts = String[]
                for i in 0:n-1
                    push!(context_parts, agent.choice_history[end-(n-i)+1] * " " * string(agent.reward_history[end-(n-i)+1]))
                end
                context = join(context_parts, " ")
                
                # Update stats
                if haskey(agent.cr_patterns, context)
                    l_count, total, _ = agent.cr_patterns[context]
                    new_l_count = l_count + (choice == "L" ? 1 : 0)
                    new_total = total + 1
                    p_value = pvalue(BinomialTest(new_l_count, new_total, 0.5))
                    agent.cr_patterns[context] = (new_l_count, new_total, p_value)
                end
            end
        end
    end
    
    # Update history
    push!(agent.choice_history, choice)
    push!(agent.reward_history, reward)
end

function choose_next_move(agent::MatchingPenniesAgent)
    # Initialize patterns if first call
    if !agent.initialized
        initialize_patterns!(agent)
    end
    
    # Not enough history
    if length(agent.choice_history) < agent.trials_back
        return rand(["L", "R"])
    end
    
    # Find the most significant pattern
    best_pvalue = 1.0
    best_bias = 0.5
    
    # Check choice-only patterns
    for n in 0:min(agent.trials_back, length(agent.choice_history))
        # Get the current context (last n choices)
        if n == 0
            context = ""
        else
            context = join(agent.choice_history[end-(n-1):end], " ")
        end
        
        # Look up the stats for this pattern
        if haskey(agent.choice_patterns, context)
            l_count, total, p_value = agent.choice_patterns[context]
            
            # Only consider patterns with enough data
            if total >= agent.trials_back
                if p_value < best_pvalue
                    best_pvalue = p_value
                    best_bias = l_count / total
                    agent.best_bias = best_bias  # Store the best bias for later use
                    agent.best_pvalue = best_pvalue  # Store the best p-value for later use
                end
            end
        end
    end
    
    # Check choice-reward patterns
    for n in 1:min(agent.trials_back, length(agent.choice_history))
        # Get the current context (last n choice-reward pairs)
        context_parts = String[]
        for i in 0:n-1
            push!(context_parts, agent.choice_history[end-(n-i)+1] * " " * string(agent.reward_history[end-(n-i)+1]))
        end
        context = join(context_parts, " ")
        
        # Look up the stats for this pattern
        if haskey(agent.cr_patterns, context)
            l_count, total, p_value = agent.cr_patterns[context]
            
            # Only consider patterns with enough data
            if total >= agent.trials_back
                if p_value < best_pvalue
                    best_pvalue = p_value
                    best_bias = l_count / total
                    agent.best_bias = best_bias  # Store the best bias for later use
                    agent.best_pvalue = best_pvalue  # Store the best p-value for later use
                end
            end
        end
    end
    
    # Make decision
    if best_pvalue > agent.alpha
        # Random choice if no significant pattern
        return rand(["L", "R"])
    else
        # Exploit pattern - choose opposite of predicted bias
        return best_bias > 0.5 ? "L" : "R"
    end
end

# Main function that's called each turn
function matching_pennies_turn(choice::String, reward::Int, t::Int)
    # Singleton pattern - use a global agent that persists between calls
    
    # If this is not the first call, update with previous choice/reward
    if t > 1
        update_agent!(mp_agent, choice, reward)
    end
    
    # Choose next move
    return choose_next_move(mp_agent)
end


# mp_agent = MatchingPenniesAgent()

# for t in 1:20
#     computer_choice = choose_next_move(mp_agent)
#     println("Turn $t: Computer chooses $computer_choice")

#     # let's choose the player's move randomly for this example
#     player_choice = rand(["L", "R"])
#     println("Player chooses $player_choice")

#     if player_choice == computer_choice
#         reward = 0  # player loses
#         println("Player loses this turn.")
#     else
#         reward = 1   # player wins
#         println("Player wins this turn.")
#     end

#     # Update the agent with the player's choice and the reward
#     update_agent!(mp_agent, player_choice, reward)
#     println("Current agent best bias: $(mp_agent.best_bias), best p-value: $(mp_agent.best_pvalue)\n")
#     println("---------------------\n")

# end