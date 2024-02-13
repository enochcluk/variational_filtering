using LinearAlgebra
using Statistics
using Random

using Distributions
#using Zygote
using ForwardDiff

m0 = 0.0
C0 = 1.0
Σ = 5.0
Γ = 1.0
A = 0.9
H = 1.0
J = 1000
J0 = 1
N = 100
d = 1

vd0 = m0 + sqrt(C0)*randn()
vd = zeros(J)
vd[1] = vd0
for j=2:J
    vd[j] = A*vd[j-1] + sqrt(Σ)*randn()
end

y = H*vd .+ sqrt(Γ)*randn(J)

function filtered(K)
    T = eltype(K)
    m = zeros(T, J)
    C = zeros(T, J)
    #m = Zygote.Buffer(zeros(J))
    m[1] = m0
    #C = Zygote.Buffer(zeros(J))
    C[1] = C0
    for j=2:J
        m[j] = (I - K*H)*A*m[j-1] + K*y[j]
        C[j] = (I - K*H)*(A*C[j-1]*A' + Σ)
    end
    return copy(m), copy(C)
end

function KL_gaussian(m1, C1, m2, C2)
    return 0.5*(log(C2/C1) - d + C1/C2 + (m2 - m1)^2/C2)
end

function log_likelihood(v, y)
    return -0.5*(sum((y[j+1] - H*v[j+1])'*inv(Γ)*(y[j+1] - H*v[j+1]) for j=J0:J-1)) - 0.5*(J-J0)*log(2*pi) - 0.5*(J-J0)*log(det(Γ))
end

function KL_sum(m, C, K, N)
    return sum(KL_gaussian(mean((I - K*H)*A*(m[j-1] + sqrt(C[j-1])*randn()) + K*y[j] for i=1:N), C[j], A*m[j-1], Σ) for j=J0+1:J)
end

function steady_state_gain()
    C = ((-1 + A^2)*Γ + H^2*Σ + sqrt(4*H^2*Γ*Σ + ((-1 + A^2)*Γ + H^2*Σ)^2))/(2*H^2)
    K = C*H'*inv(H*C*H' + Γ)
    return K
end

function var_cost(K, N)
    m, C = filtered(K)
    return KL_sum(m, C, K, N) - mean(log_likelihood(m .+ sqrt.(C).*randn(J), y) for i=1:N)
end

K_steady = steady_state_gain()

grad = x->ForwardDiff.derivative(K->var_cost(K, N), x)

# # Gradient descent
K_opt = 0.4
α = 1e-5
for i=1:100
    global K_opt -= α*grad(K_opt)#Zygote.forwarddiff(K->var_cost(K, N), K_opt)
end

print("Error in K:", abs(K_opt - K_steady))