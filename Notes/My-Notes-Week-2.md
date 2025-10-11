

# 🐾 **Binary Classification**

### 1️⃣ The Goal: Cat vs. Non-Cat

Binary classification is a supervised learning task where the goal is to predict one of two possible outcomes. The output, `y`, is always a discrete value, either **0** (negative class) or **1** (positive class)[cite: 57, 1399].

A classic example used throughout the course is building a **cat detector**.
* **Input (x):** An image.
* **Output (y):** 1 if the image is a cat, 0 if it's not a cat[cite: 62].



### 2️⃣ Representing the Input (x)
![alt text](image-5.png)

How does a computer "see" an image? An image is stored as three separate matrices for the **Red, Green, and Blue (RGB)** color channels[cite: 68, 379].

* **Pixel Intensities:** Each cell in these matrices contains a value representing pixel intensity.
* **Feature Vector:** To feed this into a neural network, we "unroll" or "reshape" these three matrices into a single, long column vector called a **feature vector, `x`**[cite: 72].
* **Input Size ($n_x$):** For a 64x64 pixel image, the resulting feature vector `x` has a dimension of `64 * 64 * 3 = 12,288`[cite: 73, 1447]. This number is the input size, denoted as $n_x$[cite: 8, 95, 1448].

![alt text](image-6.png)

### 3️⃣ Notation for the Dataset

To work with data efficiently, we use a standard set of notations[cite: 1]:

* **A single example:** is represented by a pair `(x, y)`, where `x` is the $n_x$-dimensional feature vector and `y` is the label (0 or 1)[cite: 1461].
* **Training set size:** The number of training examples is denoted by `m`[cite: 7, 93].
* **Input Matrix (X):** To process all examples at once (vectorization), we stack the individual feature vectors `x^(i)` side-by-side in columns to form a matrix `X`[cite: 15, 398].
    * The shape of `X` is $(n_x, m)$[cite: 15, 1467].
* **Label Matrix (Y):** Similarly, we stack the labels `y^(i)` side-by-side to form a row vector `Y`.
    * The shape of `Y` is $(1, m)$[cite: 17, 103, 1469].

---

