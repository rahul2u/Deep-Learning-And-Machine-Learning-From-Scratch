'''
This dataset is the current status at the beginning
of each game for the first four games in a season:
toes = current average number of toes per player
wlrec = current games won (percent)
fans = fan count (in millions

multiple input and output neural network
take input - toes,wlrec,fans
predicted or output -hurt(player),win(chances),sadness(player)

'''

 
weights = [	[0.1, 0.1,-0.3],         # hurt weight
			[0.1, 0.2,0.0],          # win weight
			[0.0, 1.3,0.1] ]         # sad weight

def sum_weight(input, weights):
	assert len(input) == len (weights)
	output = 0

	for i in range(len(input)):
		output +=  input[i] * weights[i]
	return output



def vect_mat_mul(input, weights):
	assert len(input) == len(weights)
	output = [0,0,0]

	for i in range(len(input)):
		output[i] = sum_weight(input,weights[i])
	return output
 
def neural_network(input,weights):
    predicted = vect_mat_mul(input, weights)
    return predicted


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0],wlrec[0],nfans[0]]

final_predicted = neural_network(input,weights)
print(final_predicted)