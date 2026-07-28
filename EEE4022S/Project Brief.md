***(Main in Notion)***

The Department of Astronomy at UCT would like to assemble a **two-antenna interferometer for radio astronomy imaging**. The system will _**serve as a teaching demonstrator**_ for radio astronomy courses.

Already existing at the local observatory in Observatory are two dish antennas separated by a baseline distance of about 20 metres. These operate at 1420 MHz, capturing signals hydrogen gas, and suitable for radio imaging. There is an existing data capture system that allows monitoring of the received frequency spectrum over a bandwidth of 50 MHz, _**but it does not have an interferometric mode.**_

The _**aim of this project is to assemble an interferometric data capture system that can record the RF signals from the two antennas and then to implement signal processing software to perform interferometric data processing**_. The data capture system would be located at the antenna site in Observatory, and the data should be accessible via an internet link to allow remote processing and display.

Each antenna has two outputs (dual polarisation outputs) so potentially, four channels could be simultaneously recorded, although a minimum of two (one from each antenna) would be sufficient to perform basic interferometric imaging.  
A preliminary investigation indicates that a low-cost data capture system can be assembled using software defined radio (SDR) hardware. [https://wirelesspi.com/top-5-software-defined-radios-sdr-for-rf-experimentation/](https://wirelesspi.com/top-5-software-defined-radios-sdr-for-rf-experimentation/)

ADALM-Pluto (modified to sample two channels) has been identified as a suitable platform.  
What is minimally required is to sample two channels (one from each antenna) with time synchronization. _**A minimum bandwidth of 5 MHz is desirable.**_  
The data stream needs to accessible remotely via the internet, so the SDR output (usually USB) would need to be fed to a local computer (perhaps a Raspberry Pi) which can be attached to the internet.

Tasks:

- Implementation of a suitable SDR-based solution for recoding two (possibly 4 channels).
- Record two synchronized data streams from the two antennas.
- Write software to process and display the data sets (basic spectral analysis).
- Implement and perform interferometric processing (image formation  
    involves cross correlation and Fourier transform processing)

The project has scope for more involvement in the interferometric processing. A simulator could be developed and algorithms implemented to form images. The algorithms would then be applied to the real data to form an image of an astronomical target of interest.

A budget to purchase the required SDR hardware will be provided by the UCT Department of Astronomy.