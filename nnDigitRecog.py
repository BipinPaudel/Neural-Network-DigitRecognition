import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

%matplotlib inline
#neural network class definition
class neuralNetwork:
    
    #initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,
                learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        
        #link weight matrices, wih and who
        #self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),
         #                         (self.hnodes,self.inodes))
        #self.who=np.random.normal(0.0,pow(self.onodes,-0.5),
        #                         (self.onodes,self.hnodes))
        self.wih=(np.random.rand(self.hnodes,self.inodes)-0.5)
        self.who=(np.random.rand(self.onodes,self.hnodes)-0.5)
        self.lr=learningrate
        
        #activation function
        self.activation_function=lambda x:ss.expit(x)
        pass
    
        
    # train the neural network
    def train(self,inputs_list,targets_list):
        #convert inputs list to 2d array
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(targets_list,ndmin=2).T
        
        #calculate signals into hidden layer
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        
        # output layer error is the (target-actual)
        output_errors=targets-final_outputs
        
        #hidden layer error is the ouput_errors, split by weights,
        #recombined at hidden nodes
        hidden_errors=np.dot(self.who.T,output_errors)
        
        # update the weights for the links between hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs *(1.0-final_outputs)),
                                       np.transpose(hidden_outputs))
        
        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs*(1.0-hidden_outputs)),
                                    np.transpose(inputs))
        pass
    
    #query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs=np.array(inputs_list,ndmin=2).T
        
        #calculate signals into hidden layer
        hidden_inputs=np.dot(self.wih,inputs)
        
        #calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs=np.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)
        
        return final_outputs
    
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file=open("mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()


for record in training_data_list:
    #split the record by comma
    all_values=record.split(',')
    # scale and shift the inputs
    input1=(np.asfarray(all_values[1:])/255.0*0.99) + 0.01
    #create the target output values (all 0.01,except 
    # the desired label which is 0.99)
    target1 = np.zeros(output_nodes) + 0.01
    #all_values[0] is the target label for this record
    target1[int(all_values[0])]=0.99
    n.train(input1,target1)
    pass

test_data_file=open("mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()

    
scorecard=[]
for record in test_data_list:
    all_values=record.split(',')
    #correct answer is first value
    correct_label=int(all_values[0])
    # scale and shift the inputs
    inputs=(np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    
    outputs=n.query(inputs)
    
    #the index of the highest value corresponds to the label
    label=np.argmax(outputs)
    # append correct of incorrect to list
    if(label==correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
scorecard_array=np.asfarray(scorecard)
accuracy= scorecard_array.sum()/scorecard_array.size * 100
print("Accuracy = %s " %accuracy)
