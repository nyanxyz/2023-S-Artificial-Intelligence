from pomegranate import (
    Node,
    DiscreteDistribution,
    ConditionalProbabilityTable,
    BayesianNetwork,
)

# Problem 1. building the Bayesian network diagram above
smokes = Node(DiscreteDistribution({"yes": 0.3, "no": 0.7}), name="smokes")
lung_disease = Node(
    ConditionalProbabilityTable(
        [
            ["yes", "yes", 0.1],
            ["yes", "no", 0.9],
            ["no", "yes", 0.001],
            ["no", "no", 0.999],
        ],
        [smokes.distribution],
    ),
    name="lung_disease",
)
cold = Node(DiscreteDistribution({"yes": 0.1, "no": 0.9}), name="cold")
shortness_of_breath = Node(
    ConditionalProbabilityTable(
        [
            ["yes", "yes", 0.25],
            ["yes", "no", 0.75],
            ["no", "yes", 0.01],
            ["no", "no", 0.99],
        ],
        [lung_disease.distribution],
    ),
    name="shortness_of_breath",
)
chest_pain = Node(
    ConditionalProbabilityTable(
        [
            ["yes", "yes", 0.1],
            ["yes", "no", 0.9],
            ["no", "yes", 0.01],
            ["no", "no", 0.99],
        ],
        [lung_disease.distribution],
    ),
    name="chest_pain",
)
cough = Node(
    ConditionalProbabilityTable(
        [
            ["yes", "yes", "yes", 0.75],
            ["yes", "yes", "no", 0.25],
            ["yes", "no", "yes", 0.4],
            ["yes", "no", "no", 0.6],
            ["no", "yes", "yes", 0.5],
            ["no", "yes", "no", 0.5],
            ["no", "no", "yes", 0.01],
            ["no", "no", "no", 0.99],
        ],
        [lung_disease.distribution, cold.distribution],
    ),
    name="cough",
)
fever = Node(
    ConditionalProbabilityTable(
        [
            ["yes", "yes", 0.4],
            ["yes", "no", 0.6],
            ["no", "yes", 0.02],
            ["no", "no", 0.98],
        ],
        [cold.distribution],
    ),
    name="fever",
)

model = BayesianNetwork()
model.add_states(
    smokes, lung_disease, shortness_of_breath, chest_pain, cough, fever, cold
)

model.add_edge(smokes, lung_disease)
model.add_edge(lung_disease, shortness_of_breath)
model.add_edge(lung_disease, chest_pain)
model.add_edge(lung_disease, cough)
model.add_edge(cold, cough)
model.add_edge(cold, fever)

model.bake()

# Problem 2. calculating the joint probability
probability = model.probability([["yes", "yes", "yes", "no", "yes", "no", "no"]])
print(f"Problem 2. {probability}")

print("-" * 40)

# Problem 4. calculating the probabilities of all nodes given 𝐶h𝑒𝑠𝑡𝑃𝑎𝑖𝑛 = 𝑇.
print("Problem 4.")
predictions = model.predict_proba({"chest_pain": "yes"})
for node, prediction in zip(model.states, predictions):
    if isinstance(prediction, str):
        print(f"{node.name}: {prediction}")
    else:
        print(f"{node.name}")
        for value, probability in prediction.parameters[0].items():
            print(f"    {value}: {probability:.4f}")

print("-" * 40)

# Problem 6. calculating the conditional probability 𝑃(𝐿𝑢𝑛𝑔𝐷𝑖𝑠𝑒𝑎𝑠𝑒 = 𝑇|𝑆𝑚𝑜𝑘𝑒𝑠 = 𝑇, 𝐶𝑜𝑢𝑔h = 𝑇)
predictions = model.predict_proba({"smokes": "yes", "cough": "yes"})
lung_disease_distribution = predictions[1]
print("Problem 6. {:.4f}".format(lung_disease_distribution.parameters[0]["yes"]))
