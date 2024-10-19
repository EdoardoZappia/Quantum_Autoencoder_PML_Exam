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

# Carica i dati salvati
X_train_transformed = np.load('gener/X_train_transformed.npy')
X_validation_transformed = np.load('gener/X_validation_transformed.npy')
X_test_transformed = np.load('gener/X_test_transformed.npy')
labels = np.load('gener/labels.npy')

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

# Load optimized weights
weights_path = 'gener/weights_ottimizzati.npy'
opt_weights_loaded = np.load(weights_path)

print("PESI PRECEDENTI:", opt_weights_loaded)

# Salva l'immagine del circuito
fig, ax = qml.draw_mpl(train_circuit)(opt_weights_loaded, X_train_transformed[0])
fig.savefig('gener/train_circuit.png')
plt.close(fig)

def cost_function(weights):
    probabilities = [train_circuit(weights, transaction) for transaction in X_train_transformed]
    print("Weights: ", weights)
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

print("PESI OTTIMIZZATI:", opt_weights)

# Traccia e salva la funzione di perdita durante l'addestramento
plt.figure()
plt.plot(cost_values)
plt.xlabel('Step')
plt.ylabel('Cost')
plt.title('Cost Function during Training')
plt.savefig('gener/training_cost_function.png')
plt.close()

# Reset the qubits indicated by wir to zero
def reset_to_zero(wir):
    m1 = qml.measure(wir[0])
    m2 = qml.measure(wir[1])
    qml.cond(m1, qml.PauliX)(wir[0])
    qml.cond(m2, qml.PauliX)(wir[1])

@qml.qnode(dev)
def autoencoder(opt_weights, transaction):
    qml.devices.qubit.create_initial_state([0, 1, 2, 3, 4])
    ansatz_custom_digits(transaction, n_wires_latent, n_wires_trash)
    ansatz_custom_digits(opt_weights, n_wires_latent, n_wires_trash)
    reset_to_zero([3, 4])
    qml.adjoint(ansatz_custom_digits)(opt_weights, n_wires_latent, n_wires_trash)
    return qml.density_matrix(wires=[0, 1, 2, 3, 4])

# Salva l'immagine del circuito
fig, ax = qml.draw_mpl(autoencoder)(opt_weights, X_train_transformed[0])
fig.savefig('gener/test_circuit_diagram.png')
plt.close(fig)

dev_validation = qml.device('default.qubit', wires=5)
@qml.qnode(dev)
def initial_state(transaction):
    qml.devices.qubit.create_initial_state([0, 1, 2, 3, 4])
    ansatz_custom_digits(transaction, n_wires_latent, n_wires_trash)
    return qml.density_matrix(wires=[0, 1, 2, 3, 4])

# Returns the mean of fidelity between the initial state and the autoencoder, computed on the validation set
def validation(opt_weights, X_validation_transformed):
    fid = []
    for i in X_validation_transformed:
        ae = autoencoder(opt_weights, i)
        initial = initial_state(i)
        fid.append(qml.math.fidelity(ae, initial))
    mean_fid = np.mean(fid)
    std_fid = np.std(fid)
    return mean_fid, std_fid

mean_fid, std_fid = validation(opt_weights, X_validation_transformed)

def test_supervised(opt_weights, X_test_transformed, mean_fid, std_fid, labels):
    correct_fraud = 0
    false_fraud = 0
    correct_clean = 0
    false_clean = 0
    fidelities = []
    
    threshold = mean_fid - std_fid
    
    for idx, i in enumerate(X_test_transformed):
        ae = autoencoder(opt_weights, i)
        initial = initial_state(i)
        fidelity = qml.math.fidelity(ae, initial)
        fidelities.append(fidelity)
        
        if fidelity < threshold:
            if labels[idx] == 1:
                correct_fraud += 1
            else:
                false_fraud += 1
        else:
            if labels[idx] == 1:
                false_clean += 1
            else:
                correct_clean += 1
                
    return correct_fraud, false_fraud, correct_clean, false_clean, fidelities

correct_fraud, false_fraud, correct_clean, false_clean, fidelities = test_supervised(opt_weights, X_test_transformed, mean_fid, std_fid, labels)

predictions = [1 if fidelity < (mean_fid - std_fid) else 0 for fidelity in fidelities]
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)
auc = roc_auc_score(labels, predictions)

plt.figure(figsize=(10, 6))
plt.hist([fidelities[i] for i in range(len(labels)) if labels[i] == 1], bins=30, alpha=0.5, label='Fraud')
plt.hist([fidelities[i] for i in range(len(labels)) if labels[i] == 0], bins=30, alpha=0.5, label='Legitimate')
plt.xlabel('Fidelity Score')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Distribution of Fidelity Scores')
plt.savefig('gener/distributions.png')
plt.close()

cm = confusion_matrix(labels, predictions)

plt.figure(figsize=(8, 6))
plt.matshow(cm, cmap=plt.cm.Blues, fignum=1)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
plt.yticks([0, 1], ['Legitimate', 'Fraudulent'])

for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, f'{val}', ha='center', va='center', color='red')

plt.savefig('gener/confusion_matrix.png')
plt.close()

# Save metrics
file_path = os.path.join("gener", "metrics.txt")
with open(file_path, "w") as file:
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")
    file.write(f"AUC: {auc}\n")
    file.write(f"Execution Time: {execution_time:.2f} seconds\n")
