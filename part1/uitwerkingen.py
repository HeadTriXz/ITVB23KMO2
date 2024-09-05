import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm


# OPGAVE 1
def draw_graph(data: np.ndarray) -> None:
    """
    Deze methode tekent een scatter-plot van de data die als parameter aan deze functie wordt meegegeven.

    :param data: Een twee-dimensionale matrix met in de eerste kolom de grootte van de steden,
                 in de tweede kolom de winst van de vervoerder.
    """
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Population in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()


# OPGAVE 2
def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Deze methode berekent de kosten van de huidige waarden van theta, dat wil zeggen de mate waarin de
    voorspelling (gegeven de specifieke waarde van theta) correspondeert met de werkelijke waarde (die
    is gegeven in y).

    Elk datapunt in X wordt hierin vermenigvuldigd met theta (welke dimensies hebben X en dus theta?)
    en het resultaat daarvan wordt vergeleken met de werkelijke waarde (dus met y). Het verschil tussen
    deze twee waarden wordt gekwadrateerd en het totaal van al deze kwadraten wordt gedeeld door het
    aantal data-punten om het gemiddelde te krijgen. Dit gemiddelde moet je retourneren (de variabele
    J: een getal, kortom).
    """
    m = len(y)
    predictions = np.dot(X, theta)
    errors = np.power(predictions - y, 2)

    return np.sum(errors) / (2 * m)


# OPGAVE 3a
def gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        alpha: float,
        num_iters: int
) -> tuple[np.ndarray, list[float]]:
    """
    In deze opgave wordt elke parameter van theta num_iter keer geüpdate om de optimale waarden
    voor deze parameters te vinden. Per iteratie moet je alle parameters van theta update.

    Elke parameter van theta wordt verminderd met de som van de fout van alle datapunten
    vermenigvuldigd met het datapunt zelf (zie Blackboard voor de formule die hierbij hoort).
    Deze som zelf wordt nog vermenigvuldigd met de 'learning rate' alpha.
    """
    m, n = X.shape
    costs = []

    for i in range(1, num_iters):
        predictions = np.dot(X, theta.T)
        errors = predictions - y

        theta -= (alpha / m) * np.sum(errors * X, axis=0)

        costs.append(compute_cost(X, y, theta.T))

    return theta, costs


# OPGAVE 3b
def draw_costs(data: list[float]) -> None:
    plt.plot(data)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()


# OPGAVE 4
def contour_plot(X: np.ndarray, y: np.ndarray) -> None:
    """
    Deze methode tekent een contour plot voor verschillende waarden van theta_0 en theta_1.
    De infrastructuur en algemene opzet is al gegeven; het enige wat je hoeft te doen is
    de matrix J_vals vullen met waarden die je berekent aan de hand van de methode computeCost,
    die je hierboven hebt gemaakt.

    Je moet hiervoor door de waarden van t1 en t2 itereren, en deze waarden in een ndarray
    zetten. Deze ndarray kun je vervolgens meesturen aan de functie computeCost. Bedenk of je nog een
    transformatie moet toepassen of niet. Let op: je moet computeCost zelf *niet* aanpassen.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    t1 = np.linspace(-10, 10, 100)
    t2 = np.linspace(-1, 4, 100)
    T1, T2 = np.meshgrid(t1, t2)

    J_vals = np.zeros((len(t2), len(t2)))

    for i in range(len(t1)):
        for j in range(len(t2)):
            theta = np.array([t1[i], t2[j]]).reshape(2, 1)
            J_vals[i, j] = compute_cost(X, y, theta)

    ax.plot_surface(T1, T2, J_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel(r'$\theta_0$', linespacing=3.2)
    ax.set_ylabel(r'$\theta_1$', linespacing=3.1)
    ax.set_zlabel(r'$J(\theta_0, \theta_1)$', linespacing=3.4)

    ax.dist = 10

    plt.show()
