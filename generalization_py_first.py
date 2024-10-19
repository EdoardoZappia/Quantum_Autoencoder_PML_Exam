import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import pennylane as qml
from pennylane import numpy
from scipy.optimize import minimize
import csv
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time
import os

df = pd.read_csv('creditcard.csv')

df['V29']=np.zeros((len(df['Amount']), 1))
df['V30']=np.zeros((len(df['Amount']), 1))
df['V31']=np.zeros((len(df['Amount']), 1))
df['Amount'] = np.log10(df.Amount + 0.00001)
# Split
fraud = df[df['Class'] == 1]
clean = df[df['Class'] == 0]

# Shuffle
clean = clean.sample(frac=1).reset_index(drop=True)

# training set: exlusively non-fraud transactions
X_train = clean.iloc[:1000].drop('Class', axis=1) 

# validation set: exlusively non-fraud transactions
X_validation = clean.iloc[1000:2000].drop('Class', axis=1) 

# testing  set: the remaining non-fraud + all the fraud 
fraud_test = fraud[:50]  

# Concatenazione dei dati non fraudolenti e fraudolenti
X_test = pd.concat([clean.iloc[2000:2950], fraud_test]).sample(frac=1).reset_index(drop=True) 

print(f"""Our testing set is composed as follows:

{X_test.Class.value_counts()}""")

# configure our pipeline
pipeline = Pipeline([('normalizer', Normalizer()),
                     ('scaler', MinMaxScaler())])

# get normalization parameters by fitting to the training data
pipeline.fit(X_train);

# get normalization parameters by fitting to the training data and transform the validation data with these parameters
pipeline.transform(X_validation);
X_validation_transformed = pipeline.transform(X_validation)

# transform the training with these parameters
X_train_transformed = pipeline.transform(X_train)

labels = X_test['Class']
X_test.drop('Class', axis=1, inplace=True)

pipeline.transform(X_test);
X_test_transformed = pipeline.transform(X_test)

# Salva i dati trasformati in file separati
np.save('gener/X_train_transformed.npy', X_train_transformed)
np.save('gener/X_validation_transformed.npy', X_validation_transformed)
np.save('gener/X_test_transformed.npy', X_test_transformed)
np.save('gener/labels.npy', labels)

def ansatz_custom_digits(params, n_wires_latent, n_wires_trash):
    params = qml.numpy.tensor(params, requires_grad=True)
    for j in range(6):
        for i in range(n_wires_latent+n_wires_trash):
            k = j*5+i
            qml.RY(params[k], wires=i)
        for i in range(n_wires_latent + n_wires_trash - 1):
            qml.CNOT(wires=[i, i+1])


dev = qml.device('default.qubit', wires=8)
@qml.qnode(dev)
def train_circuit(params, image):
    qml.devices.qubit.create_initial_state([0, 1, 2, 3, 4, 5, 6])
    ansatz_custom_digits(image, n_wires_latent, n_wires_trash)
    ansatz_custom_digits(params, n_wires_latent, n_wires_trash)
    swap_test(n_wires_latent, n_wires_trash)
    return qml.probs(wires=7)

def swap_test(num_latent, num_trash):
    auxiliary_qubit = num_latent + 2 * num_trash
    qml.Hadamard(auxiliary_qubit)
    for i in range(num_trash):
        qml.CSWAP(wires=[auxiliary_qubit, num_latent + i, num_latent + num_trash + i])
    qml.Hadamard(auxiliary_qubit)

n_wires_latent = 3
n_wires_trash = 2
n_wires_total = n_wires_latent + 2 * n_wires_trash +1

params = np.random.random((30,))
params = qml.numpy.tensor(params, requires_grad=True)

# Salva l'immagine del circuito
fig, ax = qml.draw_mpl(train_circuit)(params, X_train_transformed[0])
fig.savefig('gener/train_circuit.png')
plt.close(fig)

def cost_function(weights):
    probabilities = [train_circuit(weights, transaction) for transaction in X_train_transformed]
    #print(probabilities)
    cost_value = np.sum([p[1] for p in probabilities])/X_train_transformed.shape[0]
    return cost_value

cost_values = []

def callback(weights, cost_value):
    cost_values.append(cost_value)
    print(f"Step {len(cost_values)}: cost = {cost_value:.4f}, params = {weights}")


# Ottimizzazione con Adam
optimizer = AdamOptimizer(stepsize=0.1)
max_steps = 3

# trace training time
start_time = time.time()

# Ciclo di ottimizzazione
params = optimizer.step_and_cost(cost_function, params)[0]
for step in range(max_steps):
    params, cost_val = optimizer.step_and_cost(cost_function, params)
    callback(params, cost_val)

opt_weights = params

end_time = time.time()
execution_time = end_time - start_time

# Save optimized weights
weights_path = 'gener/weights_ottimizzati.npy'
np.save(weights_path, opt_weights)

print(opt_weights)