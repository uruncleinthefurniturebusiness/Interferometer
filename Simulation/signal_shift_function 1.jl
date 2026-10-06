# Script to test shifting signals in the time domain
# AJW 2022-10-10
#
# Here is some code to implement a non-integer shift via the FFT method.
# Initial testing was done line by line.
# Afterwards, I put all into a single function.
#


using FFTW
using Plots
plotly()
default(size=(800,1000)); #Plot canvas size
#default(label=""); # Turn off legends
default(ticks=:native);  #Ticks on x-axis are labelled nicely when zooming in.



########### Create a function to do the shifting job ##########

function shift_sampled_signal(x,shift_value,sample_spacing,shift_type)
# Function to shift a sampled signal (real or complex) by a non-integer amount of samples. 
# Author: AJW 2022-10-13
# x = sampled signal i.e. x[i] = x( t(i) ).  Can be real or complex.
# It is assumed that x is bandlimited and sampled above the Nyquist rate.
# shift_value = delay in whatever units (time, distance etc.)
# sample_spacing = spacing in SAME units as the shift_value
# shift_type = "circular" or "noncircular"
# returns the samplex version of "x( t-tshift )" but the end values roll around to the start
# Note: if x is of type real, then the returned signal is also type real.
# Note: If shift_value > 0 then shift is to the right (i.e. a delay)
# e.g. Delay an audio signal, sampled at 44100 Hz by 1.3 seconds using a noncircular shift:
#     shift_sampled_signal_circular(x,1.3,1/44100,"noncircular")
# e.g. Advance an audio signal, sampled at 44100 Hz by 1.3 seconds using a circular shift:
#     shift_sampled_signal_circular(x,-1.3,1/44100,"circular")


    # Do a circular shift via the frequency domain method  
    # exp(-j2pi*td*f) where td*f = (td/dt)*k*(df*dt) = (n)*k*(1/N)
    # Note: It is important that the f values used for the 2nd half of the FFT array are negative values, as this represents the negative frequencies.  This was handled in the definition of k_values

    # Create array of freq INDEX values (k-values) stored representing [0,...,fs/2,-fs/2,...,-df]
    # This is used when evaluating the exponential factor

# Note: This code works for N even or N odd:  
    N = length(x)
    if mod(N,2)==0    # case N even
        k_values = [0:N/2-1 ; -N/2:-1];    
    else   # case N odd
        k_values = [0:(N-1)/2 ; -(N-1)/2:-1]; 
    end
    
    # Calculate shift as a ratio i.e. a unitless amount 
    shift_value_ratio = shift_value/sample_spacing
      
    rotating_exp = exp.(-im*2*pi*shift_value_ratio*k_values/N) 
    
    y = ifft( fft(x) .* rotating_exp )

    # Note, y will be type complex after doing ifft, although it should have a zero imaginary part.
    
    # If a noncircular shift is desired, then zero out the rolled over part at the relevant end of the array.
    if shift_type == "noncircular" 
        
        if shift_value_ratio > 0
            end_index = Int(ceil(shift_value_ratio))
            if end_index > N
                end_index = N
            end
            y[1:end_index] .= 0  # This does not change y from complex to real.
        end
   
        if shift_value_ratio < 0   
            start_index = N-Int(ceil(abs(shift_value_ratio)))
            if start_index < 1
                start_index = 1
            end    
            y[start_index:N] .= 0
        end
        
    elseif shift_type == "circular"
        # Do nothing to y
    else
       println("Invalid parameter. Choose either `circular` or `noncircular`.")
    end
            
    # If x is type complex then return a complex y.
    # If x is not complex (i.e. real) then return the real part of y
    if occursin("Complex",string(typeof(x)))
       # x is complex, so leave y as complex.
       return y
    else
       # x is therefore real, so discard tiny imaginary part
       return real(y)
    end
       
end

##################### end of function ###########################



# Test the shifting function ####################################

println()
println()

println( "Testing function shift_sampled_signal()" )

# Create a sine wave to test the shifting
fs = 10       #  Sample rate is 10 Hz
dt = 1/fs     #  Sample spacing
t = 0:dt:10   # 0..10 seconds
N = length(t)  

f0 = 1.0   # Test frequency

x = cos.(2*pi*f0*t)  # purely real signal for testing

#x = cos.(2*pi*f0*t) .+ im*sin.(2*pi*f0*t)   # complex signal for testing


shift_value = 3.25 # in seconds
sample_spacing = 1/fs
shift_type = "noncircular"  # Alternatively use "circular"

y = shift_sampled_signal(x,shift_value,sample_spacing,shift_type)

plot_time_domain_real = plot(t,real(x),label="real(x)");
plot!(t,real(y),label="real(y)");
title!("Real parts of signals x and y")

plot_time_domain_imag = plot(t,imag(x),label="imag(x)");
plot!(t,imag(y),label="imag(y)");
title!("Imaginary parts of signals x and y")
display(plot( plot_time_domain_real, plot_time_domain_imag,layout = (2, 1)) )

@show maximum(real(y))
@show maximum(imag(y))

@show typeof(x)
@show typeof(y)

