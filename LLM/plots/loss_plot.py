import matplotlib.pyplot as plt

# Erste Loss-Werte
loss_1 = [
    1.829500, 1.753100, 1.768900, 1.806000, 1.625900, 1.503800, 1.358900, 1.186000, 1.043000, 1.004900,
    0.874500, 0.789900, 0.783300, 0.677000, 0.665900, 0.642200, 0.587000, 0.534400, 0.512700, 0.485300,
    0.485300, 0.526500, 0.485300, 0.549000, 0.515300, 0.466100, 0.502000, 0.402600, 0.408500, 0.439200,
    0.402300, 0.398700, 0.433100, 0.416800, 0.408800, 0.372000, 0.393000, 0.368500, 0.386300, 0.357700,
    0.424300, 0.380500, 0.370200, 0.386300, 0.354900, 0.381000, 0.312800, 0.351400, 0.332700, 0.353500,
    0.315900, 0.292900, 0.344500, 0.337000, 0.322300, 0.320400, 0.340700, 0.310700, 0.329600, 0.347200
]

# Zweite Loss-Werte
loss_2 = [
    1.796400, 1.702900, 1.772700, 1.734900, 1.627700, 1.448800, 1.294600, 1.193300, 1.096400, 1.008100,
    0.905000, 0.871500, 0.715900, 0.719900, 0.674400, 0.621600, 0.609700, 0.572300, 0.576000, 0.550600,
    0.508800, 0.517000, 0.481700, 0.424800, 0.662200, 0.424900, 0.464600, 0.426700, 0.428500, 0.435300,
    0.436200, 0.376300, 0.407200, 0.380500, 0.396400, 0.409900, 0.490000, 0.386900, 0.480900, 0.485800,
    0.397500, 0.495000, 0.368800, 0.358100, 0.362400, 0.704400, 0.348900, 0.328100, 0.337500, 0.342500,
    0.350300, 0.349000, 0.354100, 0.340700, 0.364500, 0.359500, 0.456800, 0.323600, 0.437700, 0.451000
]

# Epochen
epochs = list(range(1, 61))

# Plot erstellen
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_1, label="Loss 1", marker="o", linestyle="-")
plt.plot(epochs, loss_2, label="Loss 2", marker="s", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Verlauf über die Epochen")
plt.legend()
plt.grid(True)
plt.show()
