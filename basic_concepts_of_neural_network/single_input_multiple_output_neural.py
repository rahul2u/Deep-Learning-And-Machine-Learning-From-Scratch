'''
This dataset is the current status at the beginning
of each game for the first four games in a season:
toes = current average number of toes per player
wlrec = current games won (percent)
fans = fan count (in millions)

we give the win/loss records as  percentage and predict the player hurt prediction,team win prediction,fan sad prediction so i input and 3 output

'''

weights = [0.3, 0.2, 0.9]

def weight_sum(input, weights):
	output = [0,0,0]                                  
	assert len(output) == len( weights)
	for i in  range(len(weights)):
		output[i] = input * weights[i] 
	return output



def neural_network(input, weights):
	prediction = weight_sum(input, weights)
	return prediction

                   
wlrec = [0.65, 0.8, 0.8, 0.9]
input =wlrec[0]
final_prediction = neural_network(input, weights)
print(final_prediction)
                                       