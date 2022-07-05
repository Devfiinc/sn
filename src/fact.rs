
pub fn sigmoid(x : f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub fn sigmoid_derivative(x : f64) -> f64 {
    let sigmoid_x = sigmoid(x);
    sigmoid_x * (1. - sigmoid_x)
}

pub fn tanh(x : f64) -> f64 {
    x.tanh()
}

pub fn tanh_derivative(x : f64) -> f64 {
    let tanh_x = tanh(x);
    1. - tanh_x * tanh_x
}

pub fn relu(x : f64) -> f64 {
    if x < 0. {
        0.
    } else {
        x
    }
}

pub fn relu_derivative(x : f64) -> f64 {
    if x < 0. {
        0.
    } else {
        1.
    }
}

pub fn softmax(x : f64) -> f64 {
    x.exp()
}

pub fn softmax_derivative(x : f64) -> f64 {
    let softmax_x = softmax(x);
    softmax_x * (1. - softmax_x)
}

pub fn softplus(x : f64) -> f64 {
    x.ln()
}

pub fn softplus_derivative(x : f64) -> f64 {
    let softplus_x = softplus(x);
    softplus_x * (1. - softplus_x)
}

pub fn softsign(x : f64) -> f64 {
    x / (1. + x.abs())
}

pub fn softsign_derivative(x : f64) -> f64 {
    let softsign_x = softsign(x);
    softsign_x * (1. - softsign_x)
}

pub fn hard_sigmoid(x : f64) -> f64 {
    if x < 0. {
        0.
    } else if x > 1. {
        1.
    } else {
        x
    }
}

pub fn hard_sigmoid_derivative(x : f64) -> f64 {
    if x < 0. {
        0.
    } else if x > 1. {
        0.
    } else {
        1.
    }
}

pub fn exponential(x : f64) -> f64 {
    x.exp()
}

pub fn exponential_derivative(x : f64) -> f64 {
    let exponential_x = exponential(x);
    exponential_x * (1. - exponential_x)
}

pub fn linear(x : f64) -> f64 {
    x
}

pub fn linear_derivative(x : f64) -> f64 {
    1.
}

pub fn leaky_relu(x : f64) -> f64 {
    if x < 0. {
        0.01 * x
    } else {
        x
    }
}

pub fn leaky_relu_derivative(x : f64) -> f64 {
    if x < 0. {
        0.01
    } else {
        1.
    }
}

pub fn elu(x : f64) -> f64 {
    if x < 0. {
        0.01 * (x.exp() - 1.)
    } else {
        x
    }
}

pub fn elu_derivative(x : f64) -> f64 {
    if x < 0. {
        0.01 * x.exp()
    } else {
        1.
    }
}

pub fn selu(x : f64) -> f64 {
    if x < 0. {
        1.6732632423543772848170429916717 * (x.exp() - 1.)
    } else {
        x
    }
}

pub fn selu_derivative(x : f64) -> f64 {
    if x < 0. {
        1.6732632423543772848170429916717 * x.exp()
    } else {
        1.
    }
}