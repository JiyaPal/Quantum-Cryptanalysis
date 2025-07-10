import random
import mysql.connector
from mysql.connector import errorcode

# List of possible num_bits values
num_bits_list = [64, 128, 256, 512, 1024, 2048, 4096]

def generate_random_num_bits():
    return random.choice(num_bits_list)

def generate_random_error_rate():
    return random.uniform(0.01, 0.1)

def generate_basis(num_bits):
    """Generates a random list of basis choices (rectilinear or diagonal)"""
    return [random.choice(['R', 'D']) for _ in range(num_bits)]

def encode_qubit_bb84(basis, bit):
    """Encodes a classical bit based on the chosen basis in BB84 protocol"""
    if basis == 'R':
        return 'H' if bit else 'V'
    else:
        return '+' if bit else 'X'

def decode_qubit_bb84(basis, state):
    """Decodes a received quantum state based on the chosen basis in BB84 protocol"""
    if basis == 'R':
        return 1 if state == 'H' else 0
    else:
        return 1 if state == '+' else 0

def encode_qubit_b92(bit):
    """Encodes a classical bit based on the chosen basis in B92 protocol"""
    return 'H' if bit else '+'

def decode_qubit_b92(basis, state):
    """Decodes a received quantum state based on the chosen basis in B92 protocol"""
    if basis == 'R':
        return 1 if state == 'H' else 0
    else:
        return 1 if state == '+' else 0

def simulate_noise(state, error_rate):
    """Simulates noise on the quantum channel"""
    if random.random() < error_rate:
        return random.choice(['H', 'V', '+', 'X'])
    else:
        return state

def eavesdrop_bb84_intercept_resend(encoded_qubits):
    """Simulates Eve's intercept-resend attack in BB84"""
    eve_basis = generate_basis(len(encoded_qubits))
    eve_measurements = [decode_qubit_bb84(b, state) for b, state in zip(eve_basis, encoded_qubits)]
    eve_states = [encode_qubit_bb84(b, m) for b, m in zip(eve_basis, eve_measurements)]
    return eve_states, eve_measurements, eve_basis

def eavesdrop_bb84_pns(encoded_qubits):
    """Simulates Eve's Photon Number Splitting (PNS) attack in BB84"""
    eve_states = []
    eve_measurements = []
    eve_basis = []

    for state in encoded_qubits:
        if random.random() < 0.5:  # Eve successfully performs PNS
            measurement = decode_qubit_bb84('R', state)  # Assume Eve measures in the rectilinear basis
            eve_measurements.append(measurement)
            eve_basis.append('R')
            eve_states.append(encode_qubit_bb84('R', measurement))
        else:  # Eve does not perform PNS
            eve_basis.append(random.choice(['R', 'D']))
            measurement = decode_qubit_bb84(eve_basis[-1], state)
            eve_measurements.append(measurement)
            eve_states.append(encode_qubit_bb84(eve_basis[-1], measurement))

    return eve_states, eve_measurements, eve_basis

def eavesdrop_b92(encoded_qubits, attack_type):
    """Simulates Eve's attack: Intercept-Resend Attack (IRA) or Phishing Attack (PA) in B92"""
    if attack_type == 'IRA':
        eve_basis = generate_basis(len(encoded_qubits))
        eve_measurements = [decode_qubit_b92(b, state) for b, state in zip(eve_basis, encoded_qubits)]
        eve_states = [encode_qubit_b92(m) for b, m in zip(eve_basis, eve_measurements)]
    elif attack_type == 'PA':
        eve_states = ['H' if random.random() < 0.5 else '+' for _ in encoded_qubits]
        eve_measurements = [1 if state == 'H' else 0 for state in eve_states]
        eve_basis = ['R' if state == 'H' else 'D' for state in eve_states]
    return eve_states, eve_measurements, eve_basis

def save_to_db(protocol, num_bits, ber, eve_success_rate_estimate, noise_level, attack_type):
    """Saves the simulation results to the MySQL database"""
    try:
        conn = mysql.connector.connect(
            host='localhost',  
            user='root',  
            password='mysql@123#',  
            database='quantum_cryptanalysis'  
        )
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset (
                id INT AUTO_INCREMENT PRIMARY KEY,
                protocol VARCHAR(10),
                num_bits INT,
                ber FLOAT,
                eve_success_rate_estimate FLOAT,
                noise_level FLOAT,
                attack_type VARCHAR(10)
            )
        ''')

        # Insert the results
        cursor.execute('''
            INSERT INTO dataset (protocol, num_bits, bit_error_rate, eve_success_rate_estimate, noise_level, attack_type)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (protocol, num_bits, ber, eve_success_rate_estimate, noise_level, attack_type))

        conn.commit()
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        conn.close()

def bb84_simulation(num_bits, attack_type, error_rate):
    # Calculate channel error rate based on the passed error_rate
    q = error_rate

    # Alice generates random bits and basis choices
    alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
    alice_basis = generate_basis(num_bits)

    # Encode qubits
    encoded_qubits = [encode_qubit_bb84(b, a) for b, a in zip(alice_basis, alice_bits)]

    # Simulate Eve's attack
    if attack_type == 'IRA':
        eve_states, eve_measurements, eve_basis = eavesdrop_bb84_intercept_resend(encoded_qubits)
    elif attack_type == 'PNS':
        eve_states, eve_measurements, eve_basis = eavesdrop_bb84_pns(encoded_qubits)

    # Simulate noisy channel separately from Eve's attack
    received_qubits = []
    noise_count = 0
    for state in eve_states:
        noisy_state = simulate_noise(state, q)
        if noisy_state != state:
            noise_count += 1
        received_qubits.append(noisy_state)

    # Bob generates his own basis choices
    bob_basis = generate_basis(num_bits)

    # Bob decodes qubits based on his guessed basis
    bob_bits = [decode_qubit_bb84(b, state) for b, state in zip(bob_basis, received_qubits)]

    # Public channel for basis comparison
    matching_indices = [i for i in range(num_bits) if alice_basis[i] == bob_basis[i]]
    alice_matching_bits = [alice_bits[i] for i in matching_indices]
    bob_matching_bits = [bob_bits[i] for i in matching_indices]

    # Sifted key after basis comparison
    sifted_key = alice_matching_bits

    # Calculate the error rate due to Eve's attack
    eve_matching_indices = [i for i in range(num_bits) if alice_basis[i] == eve_basis[i]]
    alice_eve_matching_bits = [alice_bits[i] for i in eve_matching_indices]
    eve_matching_bits = [eve_measurements[i] for i in eve_matching_indices]

    eve_errors = sum(a != e for a, e in zip(alice_eve_matching_bits, eve_matching_bits))
    e = eve_errors / len(eve_matching_indices) if eve_matching_indices else 0

    # Calculate Bit Error Rate (BER)
    ber = (q + (1 - q) * e / 2)*100

    # Calculate Noise Level
    noise_level = (noise_count / num_bits)*100

    # Estimate Eve's success rate
    eve_success_rate_estimate = ((1 - e) * len(eve_matching_indices) / num_bits)*100

    # Save to database
    save_to_db('BB84', num_bits, ber, eve_success_rate_estimate, noise_level, attack_type)

def b92_simulation(num_bits, attack_type, error_rate):
    # Calculate channel error rate based on the passed error_rate
    q = error_rate

    # Alice generates random bits
    alice_bits = [random.randint(0, 1) for _ in range(num_bits)]

    # Encode qubits
    encoded_qubits = [encode_qubit_b92(bit) for bit in alice_bits]

    # Simulate Eve's attack
    eve_states, eve_measurements, eve_basis = eavesdrop_b92(encoded_qubits, attack_type)

    # Simulate noisy channel separately from Eve's attack
    received_qubits = []
    noise_count = 0
    for state in eve_states:
        noisy_state = simulate_noise(state, q)
        if noisy_state != state:
            noise_count += 1
        received_qubits.append(noisy_state)

    # Bob generates his own basis choices
    bob_basis = generate_basis(num_bits)

    # Bob decodes qubits based on his guessed basis
    bob_bits = [decode_qubit_b92(b, state) for b, state in zip(bob_basis, received_qubits)]

    # Public channel for basis comparison
    matching_indices = [i for i in range(num_bits) if bob_bits[i] != -1]
    alice_matching_bits = [alice_bits[i] for i in matching_indices]
    bob_matching_bits = [bob_bits[i] for i in matching_indices]

    # Sifted key after basis comparison
    sifted_key = alice_matching_bits

    # Calculate the error rate due to Eve's attack
    eve_matching_indices = [i for i in range(num_bits) if eve_measurements[i] != -1]
    eve_matching_bits = [eve_measurements[i] for i in eve_matching_indices]
    alice_eve_matching_bits = [alice_bits[i] for i in eve_matching_indices]

    eve_errors = sum(a != e for a, e in zip(alice_eve_matching_bits, eve_matching_bits))
    e = eve_errors / len(eve_matching_indices) if eve_matching_indices else 0

    # Calculate Bit Error Rate (BER)
    ber = (q + (1 - q) * e / 2)*100

    # Calculate Noise Level
    noise_level = 100*(noise_count / num_bits)

    # Estimate Eve's success rate
    eve_success_rate_estimate = ((1 - e) * len(eve_matching_indices) / num_bits)*100

    # Save to database
    save_to_db('B92', num_bits, ber, eve_success_rate_estimate, noise_level, attack_type)

def main():
    
    protocol = input("Enter protocol (BB84/B92): ").strip().upper()

    if protocol == 'BB84':
        attack_type = input("Enter Eve's attack type (IRA/PNS): ").strip().upper()
        if attack_type in ['IRA', 'PNS']:
            for i in range(1000):
                num_bits = generate_random_num_bits()
                print(i)
                error_rate = generate_random_error_rate()
                bb84_simulation(num_bits, attack_type, error_rate)
        else:
            print("Invalid attack type selected. Please choose either IRA or PNS.")
    elif protocol == 'B92':
        attack_type = input("Enter Eve's attack type (IRA/PA): ").strip().upper()
        if attack_type in ['IRA', 'PA']:
            for i in range(1000):
                num_bits = generate_random_num_bits()
                print(i)
                error_rate = generate_random_error_rate()
                b92_simulation(num_bits, attack_type, error_rate)
        else:
            print("Invalid attack type selected. Please choose either IRA or PA.")
    else:
        print("Invalid protocol selected. Please choose either BB84 or B92.")

if __name__ == "__main__":
    main()
