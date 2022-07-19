using IntervalArithmetic
## Neural Network
## Initialize Network for Training ##
# Initializing the Network is necessary for Network Training.
# The Network is created through set Input Outputs and Hidden Nodes.
# Since the Input Layer already exist, the Hidden Layer is connected
# along with the Output Layer which forms the Network.
# This Function uses Conditionals to link Connections between the Input
# Hidden and Output Layers. This Network will only have One Hidden Layer.
function initialize_network(inputs, hidden, output)
    network = Vector()
    hidden_layer = [Dict{Any, Any}("weight" => [rand(Float64) for i in 1:inputs+1]) for i in 1:hidden]
    push!(network,hidden_layer)
    output_layer = [Dict{Any, Any}("weight" => [rand(Float64) for i in 1:hidden+1]) for i in 1:output]
    push!(network,output_layer)
    return network
end
## Forward Propagation through Network ##
# The Output of a Neural Network can be calculated through Propagating
# an Input Signal. Foward Propagation is used to both Train the Network
# and Generate Predictions on a different Data Set. Forward Propagation
# goes through each Layer of each Node and Calculates the Node's Activation
# based on the Initial Inputs, these Inputs are then fed into
# the Next Layer. The Sigmoid Functions keeps the values between 1 and 0.
function activate(weights, inputs)
	# Activation is the Sum of the Weight and Input plus Bias
    activation = weights[end] # Activation is set to Bias
    for i in 1:length(weights)-1
        activation += weights[i] * inputs[i] # Sum of Weight and Input Added to Bias
    end
    return activation
end
function sigmoids(activation)
	# Sigmoid Function is 1 / 1 + e^-x
	return 1.0 / (1.0 + exp(-activation))
end
function forward_propagation(network, data)
	inputs = copy(data) # Initial Inputs
    for layer in network
		update = Vector()
		for node in layer
			# Inputs are Calculated and Stored
			activation = activate(node["weight"], inputs)
			node["output"] = [sigmoids(activation)]
			push!(update,node["output"][1])
		end
		inputs = copy(update) # Set Inputs for Next Layer
	end
	return inputs
end
## Back Propagation through Network ##
# The Error of a Neural Network can be calculated through Propagating the
# Error between the Given Outputs and Expected Output backwards through the
# Network. The Network goes from the Output Layer to the Hidden Layer while
# assigning blame. Error is calculated as the product of the Sigmoid Derivative
# of the Output and the Output subtracted by the Expected. Error Signals are
# Stored which are used in the next layer as the layers are iterated in reverse.
function transfer(output)
	# Derivative of the Sigmoid Function called here as the Sigmoid Transfer Function
	return output * (1.0 - output)
end
function backward_propagation(network, expected)
	for i in Iterators.reverse(1:length(network)) # Propagate Backwards
		layers = network[i]
        errors = Vector()
		if i != length(network)
			for j in 1:length(layers)
                error = 0.0
                for node in network[i + 1] # Relies on Previous Layer having Error Values
					error += (mid(node["weight"][j]) * mid(node["change"][1])) # Calculate Error
                end
                push!(errors,error)
            end
        else
			for j in 1:length(layers) # Layer is First in Reverse Order
				node = layers[j]
                push!(errors,mid(node["output"][1]) - mid(expected[j])) # Calculate Error
            end
        end
		for j in 1:length(layers) # Add Error to Node in Layer
			node = layers[j]
			node["change"] = [errors[j] *  mid(transfer(node["output"][1]))] # Sigmoid Derivative
		end
	end
end
## Train Networkfrom Dataset ##
# The network is trained using Stochastic Gradient Descent. Gradient Descent
# optimizes the algorithm by finding the local minimum, used to find the
# values which minimize the cost function. This is done through updating weights
# by the product of the rate error and input subtracted by the weight.
function update_network(network, data, learn_rate)
	for i in 1:length(network)
		inputs = copy(data) # Initial Data
		pop!(inputs)
		if i != 1 # Grabs Outputs from Previous Layer to be used to Calculate
			inputs = [node["output"][1] for node in network[i - 1]]
		end
		for node in network[i] # Each Node in the Layer
			for j in 1:length(inputs) # Calculates each Weight in the Node
				node["weight"][j] -= learn_rate * mid(node["change"][1]) * mid(inputs[j])
			end # Calculate the Bias in the Node
			node["weight"][end] -= learn_rate * mid(node["change"][1])
		end
	end
end
function train_network(network, dataset, learn_rate, total_epoch, total_outputs)
	for epoch in 1:total_epoch # For Each Epoch
		sum_error = 0
		for data in dataset
			outputs = forward_propagation(network, data) # Propagate Data in Network
			expects = [0.00..0.00,1.00..1.00]
			sum_error += sum([(mid(expects[i])-mid(outputs[i]))^2 for i in 1:length(expects)]) # For Debug
			backward_propagation(network, expects) # Back Propagate with Expected Values
			update_network(network, data, learn_rate) # Updates Network in Training it on the Data
		end
		if epoch == 1
			println("") # Because Atom gives me issues and Eats Up Print Statements
		end
		println("epoch = ", epoch, " error = ", sum_error)
	end
end
##
## Neural Network Driver Code ##
# This Driver Code constructs a DataSet based on the given File.
# Along with that is also declares variables which will be used
# to evaluate the Network. The Network will be scored through
# this evaluation and will display the average accuracy.
dataset = [ [0.10..0.50,0.65..0.90,0.00..0.00],
			[0.15..0.25,0.50..0.55,0.00..0.00],
			[0.05..0.90,0.50..0.90,0.00..0.00],
			[0.75..0.80,0.45..0.50,0.00..0.00],
			[0.85..0.90,0.35..0.40,0.00..0.00],
			[0.35..0.40,0.85..0.90,1.00..1.00],
			[0.45..0.50,0.85..0.90,1.00..1.00],
			[0.50..0.90,0.05..0.90,1.00..1.00],
			[0.50..0.55,0.15..0.25,1.00..1.00],
			[0.65..0.90,0.10..0.50,1.00..1.00] ]
inputs = length(dataset[1]) -  1
outputs = length(unique([data[end] for data in dataset]))
network = initialize_network(inputs, 2, outputs)
train_network(network, dataset, 0.5, 100, outputs)
for layer in  network
	for node in layer
		println(node)
	end
end
##
## Finished 4/8/2022 at 11:23am
# https://www.ibm.com/cloud/learn/neural-networks
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
