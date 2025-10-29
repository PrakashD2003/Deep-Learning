---
### 📝 **Week 2 Programming Assignment: Logistic Regression with a Neural Network Mindset**

This assignment guides you through building an image classification model to recognize cats versus non-cats. While it uses logistic regression, it introduces the core concepts and workflow with a neural network perspective, laying the foundation for more complex models later.

---
### 1️⃣ Problem Overview and Dataset

* **Goal:** Build a binary classifier to distinguish cat images (label `y=1`) from non-cat images (label `y=0`).
* **Dataset:** Provided in an H5 file (`data.h5`).
    * **Training Set:** `m_train` images (`train_set_x_orig`) with labels (`train_set_y`).
    * **Test Set:** `m_test` images (`test_set_x_orig`) with labels (`test_set_y`).
    * **Image Shape:** Each image is `(num_px, num_px, 3)`, where 3 represents the RGB color channels.
    * **Labels:** `train_set_y` and `test_set_y` are row vectors of shape `(1, m_train)` and `(1, m_test)` respectively.

---
### 2️⃣ Data Preprocessing

Before feeding data into the model, several preprocessing steps are essential:

1.  **Understand Dimensions:** First, determine key dimensions:
    * `m_train`: Number of training examples.
    * `m_test`: Number of test examples.
    * `num_px`: Height/width of each image.
2.  **Flatten Images:** Reshape the images from `(num_px, num_px, 3)` into single column vectors of shape `(num_px * num_px * 3, 1)`. The entire training and test datasets become matrices where each column is a flattened image.
    * `train_set_x_flatten` shape: `(num_px * num_px * 3, m_train)`
    * `test_set_x_flatten` shape: `(num_px * num_px * 3, m_test)`
    * **Code:** `X_flatten = X.reshape(X.shape[0], -1).T`
3.  **Standardize Data:** For image datasets, a simple and effective standardization method is to divide all pixel values by 255 (the maximum pixel intensity). This ensures all features have a similar range, preventing potential issues during training.
    * **Code:** `train_set_x = train_set_x_flatten / 255.`

---
### 3️⃣ Algorithm Architecture (Neural Network Mindset)

Logistic regression can be viewed as a very simple neural network with a single neuron using a sigmoid activation function.


**Mathematical Steps for one example x⁽ⁱ⁾:**
1.  **Linear Step:** `z⁽ⁱ⁾ = wᵀx⁽ⁱ⁾ + b`
2.  **Activation:** `a⁽ⁱ⁾ = sigmoid(z⁽ⁱ⁾) = ŷ⁽ⁱ⁾` (This `a` is the prediction)
3.  **Loss Function (for one example):** `L(a⁽ⁱ⁾, y⁽ⁱ⁾) = -[ y⁽ⁱ⁾ log(a⁽ⁱ⁾) + (1 - y⁽ⁱ⁾) log(1 - a⁽ⁱ⁾) ]`
4.  **Cost Function (average over all m examples):** `J = (1/m) * Σ L(a⁽ⁱ⁾, y⁽ⁱ⁾)`

**Goal:** Find the parameters `w` and `b` that minimize the cost function `J`.

---
### 4️⃣ Building the Algorithm Components

The assignment breaks the model construction into helper functions:

1.  **`sigmoid(z)`:**
    * **Purpose:** Computes the sigmoid activation function `1 / (1 + e⁻ᶻ)`.
    * **Implementation:** Uses `np.exp()`.
2.  **`initialize_with_zeros(dim)`:**
    * **Purpose:** Initializes the weight vector `w` as a vector of zeros with shape `(dim, 1)` and the bias `b` to 0.0. `dim` corresponds to the number of features (`num_px * num_px * 3`).
    * **Implementation:** Uses `np.zeros()`.
3.  **`propagate(w, b, X, Y)`:**
    * **Purpose:** Implements the core forward and backward propagation steps for one iteration.
    * **Forward Propagation:**
        * Calculates activations `A = sigmoid(wᵀX + b)` using vectorized operations (`np.dot`, broadcasting for `b`).
        * Calculates the cost `J` using the vectorized formula involving `np.sum` and `np.log`.
    * **Backward Propagation:**
        * Calculates the gradients based on the derived formulas:
            * `dw = (1/m) * X * (A - Y)ᵀ`
            * `db = (1/m) * sum(A - Y)`
        * **Implementation:** Uses `np.dot`, `np.sum`, and matrix transposition (`.T`).
    * **Returns:** `grads` (a dictionary containing `dw` and `db`) and `cost`.
4.  **`optimize(w, b, X, Y, num_iterations, learning_rate, print_cost)`:**
    * **Purpose:** Performs the gradient descent optimization loop.
    * **Steps:**
        * Iterates `num_iterations` times.
        * In each iteration:
            * Calls `propagate()` to get the current cost and gradients (`dw`, `db`).
            * Updates parameters using the gradient descent rule:
                * `w = w - learning_rate * dw`
                * `b = b - learning_rate * db`
        * Optionally records and prints the cost periodically.
    * **Returns:** `params` (final `w`, `b`), `grads` (last gradients), `costs` (list of costs recorded).
5.  **`predict(w, b, X)`:**
    * **Purpose:** Uses the learned parameters `w` and `b` to make predictions (0 or 1) on a given dataset `X`.
    * **Steps:**
        * Calculate activations `A = sigmoid(wᵀX + b)`.
        * Convert probabilities in `A` to predictions: `Y_prediction = 1` if `A > 0.5`, else `0`.
    * **Implementation:** Can use a loop or vectorized comparison `(A > 0.5).astype(int)`.
    * **Returns:** `Y_prediction`.

---
### 5️⃣ Assembling the `model()` Function

The `model()` function orchestrates the entire process:

1.  Initializes parameters (`w`, `b`) using `initialize_with_zeros()`.
2.  Runs gradient descent using `optimize()` to learn the optimal parameters.
3.  Retrieves the learned `w` and `b` from the results of `optimize()`.
4.  Makes predictions on both the training set (`X_train`) and test set (`X_test`) using `predict()`.
5.  Calculates and prints the training and test accuracies.
6.  Returns a dictionary `d` containing all relevant information (costs, predictions, learned parameters, hyperparameters).

---
### 6️⃣ Training and Analysis

* **Execution:** Running the `model()` function trains the logistic regression classifier.
* **Results:** The output shows the cost decreasing over iterations and the final training/test accuracies.
* **Interpretation:**
    * **High Training Accuracy:** Indicates the model has sufficient capacity to fit the training data.
    * **Test Accuracy:** Measures how well the model generalizes to unseen data.
    * **Overfitting:** If training accuracy is much higher than test accuracy, the model is likely overfitting (learning the training data too well, including noise, and not generalizing).

---
### 7️⃣ Learning Rate (Hyperparameter Tuning)

* **Importance:** The choice of learning rate (`α`) significantly impacts gradient descent.
* **Too Large:** Cost may oscillate or even diverge.
* **Too Small:** Convergence may be very slow, requiring many iterations.
* **Analysis:** The assignment demonstrates plotting learning curves (cost vs. iterations) for different learning rates to find a good value. A well-chosen rate shows a steadily decreasing cost.

---
### 💡 Key Takeaways from the Assignment

1.  **Preprocessing is Crucial:** Reshaping and standardizing data are vital steps.
2.  **Modular Design:** Building complex models by implementing smaller, reusable functions (`initialize`, `propagate`, `optimize`, `predict`) is a key practice.
3.  **Vectorization is Essential:** Eliminating `for` loops using NumPy operations (`np.dot`, `np.sum`, broadcasting) dramatically speeds up computation, especially on large datasets.
4.  **Neural Network Mindset:** Logistic regression, while simple, shares the fundamental structure (parameters, forward pass, activation, loss, backward pass, optimization) of more complex neural networks.
5.  **Hyperparameter Tuning:** Parameters like the learning rate are not learned from data but set beforehand and significantly influence model performance.