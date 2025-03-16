import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from RS_function import RS_function
import pandas as pd
import scipy.integrate as it
from scipy.fft import fft, fftfreq
import zipfile, io
import scipy



def chunkstring(string, length):
    return (float(string[0+i:length+i]) for i in range(0, len(string), length))

def readchunk(f, numofLines):
    x=[]
    for line in islice(f, 0,  numofLines):
        x = x + list(chunkstring(line[0:len(line)-1],10))
    #print(x)
    return x

def lines(points):
    if points % 8 == 0:
        nLines = int(points/8) 
    else:
        nLines = int(points/8)+1
    return nLines

def scaleValue(units):
    if units =="cm/sec2":
        return 1/980.665
    elif units == "cm/s^2":
        return 1/980.665
    else:
        return 1.0
    
def maxaccel(x, t):
    ymax = max(x)
    xpos = x.index(ymax); xmax = t[xpos]
    return [xmax, ymax]

def minaccel(x, t):
    ymin = min(x)
    xpos = x.index(ymin); xmin = t[xpos]
    return [xmin, ymin]

def startlimAccel():
    a1 = next(i for i, x in enumerate(accel1[0]) if abs(x) >5)
    startTime  = max(a1*dtAccel1[0]- 2, 0)
    return round(startTime,2)

def endlimAccel():
    a1 = next(i for i, x in reversed(list(enumerate(accel1[0]))) if abs(x) >5)
    endTime  = max(a1*dtAccel1[0]+ 2, 0)
    return round(endTime,2)

def saveFile():
    T1 = np.arange(0.0,numofPointsAccel1[0]*dtAccel1[0], dtAccel1[0])
    textstring=""
    j=0
    textstring += "Time(sec)"
    for i in range(numofChansRead):
        textstring += ", " + nameCh1[i].replace(",","_").replace(" ", "_")
    textstring += "\n"
    index = len(T1)
    while j < index:
        textstring += str(round(T1[j],3))
        for i in range(numofChansRead):
            textstring += ", " + str(scaledAccel1[i][j]) 
        textstring += "\n"
        j+= 1
    return (textstring)

@st.cache_data
def plotchannel(i,j,ax):
    if doption =="Disp":
        rT='Disp (cm)'
        yV1 = displ1[i]
    elif doption =="Vel":
        rT ='Vel (cm/sec)'
        yV1 = vel1[i]
    else:
        rT ='Accel (g)'
        yV1 = scaledAccel1[i]

    if doption in ["Accel", "Vel", "Disp"]:
        T1 = np.arange(0.0,numofPointsAccel1[i]*dtAccel1[i], dtAccel1[i])
        ax[j].plot(T1,yV1)
        ax[j].text(0.99, 0.97, nameCh1[i] + "; " + location[i] + "; " + rcdTime[i], horizontalalignment='right', verticalalignment='top', fontsize=10, color ='Black',transform=ax[j].transAxes)
        ax[j].set_xlabel('Secs')
        amax=maxaccel(yV1, T1)
        ax[j].annotate(str(np.round(amax[1],3)), xy=(amax[0], amax[1]), xytext=(amax[0], amax[1]))
        amin=minaccel(yV1, T1)
        ax[j].annotate(str(np.round(amin[1],3)), xy=(amin[0], amin[1]), xytext=(amin[0], amin[1]), verticalalignment='top')
        ax[j].set_xlim(starttime,endtime)
        ax[j].set_ylabel(rT)
    elif doption =="FFT for Accel":
        N = len(scaledAccel1[i])
        yf = fft(scaledAccel1[i])
        xf = fftfreq(N, dtAccel1[i])[:N//2]
        ax[j].plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax[j].set_xlim(0,10)
        ax[j].set_xlabel('Hz')
        ax[j].text(0.99, 0.97, nameCh1[i] + "; " + location[i] + "; " + rcdTime[i], horizontalalignment='right', verticalalignment='top', fontsize=10, color ='Black',transform=ax[j].transAxes)
        amax=[xf[np.argmax(abs(yf))], 2.0/N *max(abs(yf))]; ax[j].annotate(str(np.round(amax[0],3)) +"Hz", xy=(amax[0], amax[1]), xytext=(amax[0], amax[1]))
    elif doption =="Floor Spectra":
        tT = np.concatenate( (np.arange(0.05, 0.1, 0.005) , np.arange (0.1, 0.5, 0.01) , np.arange (0.5, 1, 0.02) , np.arange (1, 6, 0.05) ) ) # Time vector for the spectral response
        Sfin=[]
        xi = 0.05
        df = 1.0/dtAccel1[i]
        Sfin= RS_function(accel1[i][int(starttime/dtAccel1[i]):int(endtime/dtAccel1[i])], df, tT, xi, Resp_type = "SA")
        sAll=Sfin[0,:]*scaleValue(unitsAccel1[i])
        print(df)
        amax=[tT[np.argmax(abs(sAll))], max(abs(sAll))]
        labl = "Damping = "+ str(round(xi,3))+ ": Max at "+ str(round(amax[0],3)) +"sec, "+str(round(amax[1],2))
        ax[j].plot(tT,sAll,linewidth=1.0) 
        ax[j].annotate(labl, xy=(amax[0], amax[1]), xytext=(amax[0], amax[1]))
        ax[j].set_ylabel("Sa (g)")
    ax[j].grid()
    return(0)


def plotAll():

    noUpChns = 0; noNSChns = 0; noEWChns = 0
    for i in range(numofChansRead):
        if "Up" in nameCh1[i] or "UP" in nameCh1[i]:
            noUpChns += 1
        elif "360" in nameCh1[i] or " 0 Deg" in nameCh1[i] or " 0 DEG" in nameCh1[i]:
            noNSChns += 1
        elif "90" in nameCh1[i] or "270" in nameCh1[i]:
            noEWChns += 1

    #Up Channnel plots
    if noUpChns > 0:
        noSubplotsRows = noUpChns 
        noSubplotsCols = 1
        subplotCounter = 1
        #print(noUpChns)

        fig, ax = plt.subplots(noSubplotsRows,noSubplotsCols, sharex='col', sharey='col')
        fig.set_figheight(height*noUpChns)
        fig.set_figwidth(width)
        st.write('Vertical Motion Channels')
        fig.tight_layout()

        j=0
        for i in reversed(range(numofChansRead)):
            if "Up" in nameCh1[i] or "UP" in nameCh1[i]:
                plotchannel(i,j,ax)
                j+=1
        ax[0].set_ylim(ax[0].get_ylim()[0]*1.4,ax[0].get_ylim()[1]*1.4)
        st.pyplot(fig) 


    #NS Channnel plots

    noSubplotsRows = noNSChns
    noSubplotsCols = 1
    subplotCounter = 1

    fig2, ax2 = plt.subplots(noSubplotsRows,noSubplotsCols, sharex='col', sharey='col')
    fig2.set_figheight(height*noNSChns)
    fig2.set_figwidth(width)
    st.write('NS Channels')
    fig2.tight_layout()
    #figfft2.suptitle('Fourier Transform of NS Channels', fontsize=11)
    #figfft2.canvas.manager.set_window_title('Fourier Transform of NS Channels')

    j=0
    for i in reversed(range(numofChansRead)):
        if "360" in nameCh1[i] or " 0 Deg" in nameCh1[i] or " 0 DEG" in nameCh1[i]:
            plotchannel(i,j,ax2)
            j+=1
    ax2[0].set_ylim(ax2[0].get_ylim()[0]*1.4,ax2[0].get_ylim()[1]*1.4)
    st.pyplot(fig2)
    
    #EW Channnel plots

    noSubplotsRows = noEWChns
    noSubplotsCols = 1
    subplotCounter = 1

    fig3, ax3 = plt.subplots(noSubplotsRows,noSubplotsCols, sharex='col', sharey='col')
    fig3.set_figheight(height*noEWChns)
    fig3.set_figwidth(width)
    st.write('EW Channels')
    fig3.tight_layout()

    #figfft3.suptitle('Fourier Transform of EW Channels', fontsize=11)
    #figfft3.canvas.manager.set_window_title('Fourier Transform of EW Channels')


    j=0
    for i in reversed(range(numofChansRead)):
        if "90" in nameCh1[i] or "270" in nameCh1[i]:
            plotchannel(i,j,ax3)
            j+=1
    ax3[0].set_ylim(ax3[0].get_ylim()[0]*1.4,ax3[0].get_ylim()[1]*1.4)
    st.pyplot(fig3)
    return(0)

def readFile():
    global filenames, rcdTime
    global latitude, longitude
    global nameCh1,dtAccel1,dtDispl1,dtVel1
    global numofPointsAccel1,location,numofPointsVel1,numofPointsDispl1
    global T1
    global scaledAccel1, accel1, vel1, displ1
    global EOF,numofChansRead
    global unitsAccel1, unitsVel1, unitsDispl1, maxFFTylim, stationNo

    latitude=[]; longitude=[];accel1=[];vel1=[];displ1=[];scaledAccel1=[];nameCh1=[]; location=[]
    numofPointsAccel1=[]; dtAccel1=[]; unitsAccel1=[]
    numofPointsVel1=[]; dtVel1=[]; unitsVel1=[]
    numofPointsDispl1=[]; dtDispl1=[]; unitsDispl1=[]
    rcdTime=[]

    numofChansRead = 0
    if filenames != None:
        archive = zipfile.ZipFile(filenames, 'r')
        flist = archive.namelist()
        # stationNo = flist[0][flist[0].find("_ce")+1:-5]
        # #print(stationNo)
        f=io.BytesIO(archive.read(flist[0]))
        archive2 = zipfile.ZipFile(f, 'r')
        flist2 = archive2.namelist()
         #print( len(flist2))
        for index, value in enumerate(flist2):
            #print(flist2[index])
            if value[-3:] == ".v2" or  value[-3:] == ".V2":
                numofChansRead+=1
                f=io.TextIOWrapper(io.BytesIO(archive2.read(value)))

                for line in islice(f, 2, 3):
                    rcdTime.append(line[line.find("Rcrd of") + 7:43].strip())
                    #print(rcdTime[-1])

                for line in islice(f, 2, 3):    
                    stationNo = line[11:17].strip()
                    latlong= line[17:40].strip()
                    latitude.append(float(latlong[:latlong.find(",")-1]))
                    if latlong[len(latlong)-1:len(latlong)]=="W":
                        longitude.append(float("-"+latlong[latlong.find(",")+1: len(latlong)-1].strip()))
                    else:
                        longitude.append(float(latlong[latlong.find(",")+1: len(latlong)-1].strip()))
                    #print(latitude[index], longitude[index])

                for line in islice(f, 1, 2):
                    location.append(line[line.find("Location: ") + 10:].strip())
                    #print(location[-1])

                for line in islice(f, 15, 16):
                    nameCh1.append(line[26:].strip())
                for line in islice(f, 0, 1):
                    nameCh1[-1]=nameCh1[-1] + line.strip()

                for line in islice(f, 20, 21):
                    #print(line)
                    line = line.lower()
                    numofPointsAccel1.append(int(line[0: line.find("points")].strip()))
                    dtAccel1.append(float(line[line.find("at ") + 3: line.find(" sec")].strip()))
                    unitsAccel1.append(line[line.find(", in") + 4: line.find(". (")].strip())
                numofLines=(lines(numofPointsAccel1[-1]))
                accel1.append(readchunk(f,numofLines))
                

                for line in islice(f, 0,1):
                    #print(line)
                    line = line.lower()
                    numofPointsVel1.append(int(line[0: line.find("points")].strip()))
                    dtVel1.append(float(line[line.find("at ") + 3: line.find(" sec")].strip()))
                    unitsVel1.append(line[line.find(", in") + 4: line.find(".  (")].strip())
                numofLines = lines(numofPointsVel1[-1])
                vel1.append(readchunk(f,numofLines))

                for line in islice(f, 0,1):
                    #print(line)
                    line = line.lower()
                    numofPointsDispl1.append(int(line[0: line.find("points")].strip()))
                    dtDispl1.append(float(line[line.find("at ") + 3: line.find(" sec")].strip()))
                    unitsDispl1.append(line[line.find(", in") + 4: line.find(".   ")].strip())
                numofLines = lines(numofPointsDispl1[-1])
                displ1.append(readchunk(f,numofLines))

                
                scale = scaleValue(unitsAccel1[-1]) 
                scaledAccel1.append([value*scale for value in accel1[-1]])
    if numofChansRead == 1:
        st.write("File is probably a free-field record and not a building record. Please upload a building record.")
        st.stop()
    return(0)

def plottransfer():
    inputChn =chanList.index(inChn)
    outputChn = chanList.index(outChn)
    # print(inputChn)
    # print(outputChn)

    Ni = len(scaledAccel1[inputChn])
    yfi = fft(scaledAccel1[inputChn])
    xfi = fftfreq(Ni, dtAccel1[inputChn])[:Ni//2]

    No = len(scaledAccel1[outputChn])
    yfo = fft(scaledAccel1[outputChn])
    xfo = fftfreq(No, dtAccel1[outputChn])[:No//2]

    tF = np.divide(yfo,yfi)

    fig4, axa = plt.subplots(3,1, sharex='col', sharey='row')
    fig4.set_figheight(height*3)
    fig4.set_figwidth(width)
    axa[0].plot(xfi,  np.abs(tF[0:No//2]))
    axa[0].set_xlim(0.0,10)
    axa[0].set_ylim(0.0,None)
    axa[0].set_ylabel('Amplitude')
    axa[0].grid()

    axa[1].plot(xfi,  np.angle(tF[0:No//2],deg=True))
    axa[1].set_ylabel('Phase Angle')
    axa[1].grid()

    f, Cxy = scipy.signal.coherence(scaledAccel1[inputChn], scaledAccel1[outputChn], 1/dtAccel1[inputChn])
    axa[2].semilogy(f, Cxy)
    axa[2].set_xlabel('frequency [Hz]')
    axa[2].set_ylabel('Coherence')
    axa[2].grid()
    return(fig4)


st.title("Vizualize/Plot Recorded Earthquake Building Motions")
st.write("V2/V2c files are building earthquake records that can be downloaded from Center for Earthquake Engineering Strong Motion CESMD webiste.  Download one free-field record at a time and do not unzip.")
st.write("https://www.strongmotioncenter.org/")
st.write("This app helps read the file and show the recording and create spectra from the recordings")
filenames=st.file_uploader("Upload V2/V2c zip file",type=[ "zip"])    
if filenames != None:    
    readFile()
    st.header(rcdTime[0])
    df = pd.DataFrame({"lat":[float(latitude[0])], "lon":[float(longitude[0])]})
    st.map(df)   
    c1, c2 =st.columns(2)
    with c1:
        st.link_button("See Instrument Details", 'https://www.strongmotioncenter.org/cgi-bin/CESMD/stationhtml.pl?stationID=CE'+stationNo+'&network=CGS')  
    with c2:
        st.link_button("See location of instrument in Google Maps", 'http://www.google.com/maps/place/'+ str(latitude[0]) +','+str(longitude[0])+'/@'+ str(latitude[0]) +','+str(longitude[0])+',12z')

    st.subheader("Recorded Values")
    st.write("Read in memory :\n"+ str(filenames.name))
    st.write("Number of channels read: " + str(numofChansRead))
    values = st.sidebar.slider("Select range of times to use", 0.0, dtAccel1[0]*numofPointsAccel1[0], (startlimAccel(), endlimAccel()), step= 0.1)
    st.sidebar.caption("*Range autoselected using a trigger of 0.005g")
    starttime, endtime = values
    width = st.sidebar.slider("plot width", 10, 20, 10)
    height = st.sidebar.slider("plot height", 1, 10, 3)
    doption = st.selectbox("Plot",("Accel", "Vel", "Disp", "FFT for Accel","Floor Spectra"),)
    if doption != None:
        st.write("Plotting " + doption + " for all channels")
        plotAll()

    st.subheader("Transfer Function")
    st.write("Select the input and output channels to plot the transfer function")
    chanList=[i + ";" + j for i, j in zip(nameCh1, location)]
    inChn = st.selectbox("Select Input Chan for transfer function", chanList)
    outChn = st.selectbox("Select output Chan for transfer function", chanList)
    if inChn == outChn:
        st.write("Input and Output channels are the same. Please select different channels.")
    else:   
        if st.button("Plot Transfer Function"):
            fig =plottransfer()
            st.pyplot(fig)
    
    st.subheader("Download Accelerations")
    text_contents = saveFile()
    st.download_button("Save Acceleration file", text_contents, file_name="accelerations.csv",mime="text/csv",)