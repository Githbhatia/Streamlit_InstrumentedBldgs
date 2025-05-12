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
from streamlit_sortables import sort_items
import math


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

def startlimAccel(z):
    startTime = dtAccel1[0]*numofPointsAccel1[0]/2
    for j in range(numofChansRead):
        a1 = next(i for i, x in enumerate(accel1[j]) if abs(x) >z)
        startTime  = min(a1*dtAccel1[j]- 2, startTime)
    return round(startTime,2)

def endlimAccel(z):
    endTime = dtAccel1[0]*numofPointsAccel1[0]/2
    for j in range(numofChansRead):
        a1 = next(i for i, x in reversed(list(enumerate(accel1[j]))) if abs(x) >z)
        endTime  = max(a1*dtAccel1[j]+ 2, endTime)
    return round(endTime,2)

def saveFile(avd):
    T1 = np.arange(0.0,numofPointsAccel1[0]*dtAccel1[0], dtAccel1[0])
    textstring=""
    j=0
    textstring += "Time(sec)"
    for i in range(numofChansRead):
        textstring += ", " + nameCh1[i].replace(",","_").replace(" ", "_") + ":" + location[i].replace(",","_").replace(" ", "_")
    textstring += "\n"
    index = len(T1)
    while j < index:
        textstring += str(round(T1[j],3))
        for i in range(numofChansRead):
            if avd == "Accel":
                textstring += ", " + str(scaledAccel1[i][j])
            elif avd == "Vel":
                textstring += ", " + str(vel1[i][j])
            else:
                textstring += ", " + str(displ1[i][j])
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
        ax[j].set_xlim(0,maxFFTylim)
        ax[j].set_xlabel('Hz')
        ax[j].text(0.99, 0.97, nameCh1[i] + "; " + location[i] + "; " + rcdTime[i], horizontalalignment='right', verticalalignment='top', fontsize=10, color ='Black',transform=ax[j].transAxes)
        amax=[xf[np.argmax(abs(yf))], 2.0/N *max(abs(yf))]; ax[j].annotate(str(np.round(amax[0],3)) +"Hz", xy=(amax[0], amax[1]), xytext=(amax[0], amax[1]))
    elif doption =="Floor Spectra":
        tT = np.concatenate( (np.arange(0.05, 0.1, 0.005) , np.arange (0.1, 0.5, 0.01) , np.arange (0.5, 1, 0.02) , np.arange (1, eT, 0.05) ) ) # Time vector for the spectral response
        Sfin=[]
        df = 1.0/dtAccel1[i]
        Sfin= RS_function(accel1[i][int(starttime/dtAccel1[i]):int(endtime/dtAccel1[i])], df, tT, xi, Resp_type = "SA")
        sAll=Sfin[0,:]*scaleValue(unitsAccel1[i])
        amax=[tT[np.argmax(abs(sAll))], max(abs(sAll))]
        labl = "Damping = "+ str(round(xi,3))+ ": Max at "+ str(round(amax[0],3)) +"sec, "+str(round(amax[1],2))
        ax[j].plot(tT,sAll,linewidth=1.0) 
        ax[j].annotate(labl, xy=(amax[0], amax[1]), xytext=(amax[0], amax[1]))
        ax[j].set_ylabel("Sa (g)")
        ax[j].text(0.99, 0.97, nameCh1[i] + "; " + location[i] + "; " + rcdTime[i], horizontalalignment='right', verticalalignment='top', fontsize=10, color ='Black',transform=ax[j].transAxes)
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
    #print(noUpChns,noNSChns, noEWChns)

    #Up Channnel plots
    if noUpChns > 0:

        noSubplotsRows = noUpChns 
        noSubplotsCols = 1
        subplotCounter = 1
        #print(noUpChns)
        if noUpChns == 1:
            noSubplotsRows = 2
        fig, ax = plt.subplots(noSubplotsRows,noSubplotsCols, sharex='col', sharey='col')
        fig.set_figheight(height*noUpChns)
        if noUpChns == 1:
            fig.set_figheight(height*2)
        fig.set_figwidth(width)
        st.write('Vertical Motion Channels')
        fig.tight_layout()

        j=0
        for i in reversed(range(numofChansRead)):
            if "Up" in nameCh1[revchan[i]] or "UP" in nameCh1[revchan[i]]:
                plotchannel(revchan[i],j,ax)
                j+=1
        ax[0].set_ylim(ax[0].get_ylim()[0]*1.4,ax[0].get_ylim()[1]*1.4)
        if noUpChns == 1:
            ax[1].axis('off')

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
        if "360" in nameCh1[revchan[i]] or " 0 Deg" in nameCh1[revchan[i]] or " 0 DEG" in nameCh1[revchan[i]]:
            plotchannel(revchan[i],j,ax2)
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
        if "90" in nameCh1[revchan[i]] or "270" in nameCh1[revchan[i]]:
            plotchannel(revchan[i],j,ax3)
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
    global unitsAccel1, unitsVel1, unitsDispl1, maxFFTylim, stationNo,stationD

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
        baseName = flist2[0][:(flist2[0]).find(".")]
        #print(baseName)
        if (baseName + ".v2" in flist2) and (baseName + ".v2c" in flist2):
            #print("in")
            f = io.TextIOWrapper(io.BytesIO(archive2.read(baseName + ".v2")))
            f.seek(0,2)
            position = f.tell()
            while position > 0:
                position -= 1
                f.seek(position)
                thisLine = f.readline()
                if "End of data for channel" in thisLine:
                    last_line = thisLine
                    break
            numofChansRead= int(last_line[last_line.find("channel")+7:last_line.find("---")].strip())
            # print(numofChansRead)
            f.seek(0,0)
            for i in range(0,numofChansRead):
                    #print(i)
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
                        stationD = line[26:line.find("Chan")].strip()
                    for line in islice(f, 0, 1):
                        nameCh1[-1]=nameCh1[-1] + line.strip()
                        #print(nameCh1[-1])

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
                    f.readline()

        else:
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
                        stationD = line[26:line.find("Chan")].strip()
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
    st.write("Using Input Channel = " + chanList[inputChn])
    st.write("Using Output Channel = " +chanList[outputChn])
    print(outputChn)

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
    axa[0].semilogy(xfi,  np.abs(tF[0:No//2]))
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

st.logo("HXBLogo.png", size="large")
st.title("Vizualize/Plot Recorded Earthquake Building Motions")
st.write("V2 files are building earthquake records that can be downloaded from Center for Earthquake Engineering Strong Motion CESMD webiste.  Download one building record at a time and do not unzip.")
st.write("https://www.strongmotioncenter.org/")
st.write("Also see https://hcai.ca.gov/facilities/building-safety/facility-detail/ for instrumented hospital buildings")
st.write("This app helps read the file and show the recording and create floor spectra from the recordings")
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
    trigger = 5.0
    for i in range(numofChansRead):
        trigger = min(abs(max(accel1[i], key=abs))/10,trigger)
    values = st.sidebar.slider("Select range of times to use", 0.0, dtAccel1[0]*numofPointsAccel1[0], (startlimAccel(trigger), endlimAccel(trigger)), step= 0.1)
    st.sidebar.caption("*Range autoselected using a trigger of " + str(round(trigger*scaleValue(unitsAccel1[0]),3)) + "g")
    starttime, endtime = values
    width = st.sidebar.slider("plot width", 10, 20, 10)
    height = st.sidebar.slider("plot height", 1, 10, 3)

    st.write("station: " + stationD)
    chanList2=[i + ";" + j for i, j in zip(nameCh1, location)]
    chanList = [i.replace(stationD,"") for i in chanList2]
    sorted_items = chanList.copy()
    sorted_items.reverse()
    sorted_items2 = chanList.copy()
    sorted_items2.reverse()
    rearrange = st.checkbox("Rearrange channels to change display order?", value=False)
    if rearrange:
        st.write("Drag and drop to rearrange channels (Suggested order: In descending order of floors with roof at top and basement at bottom)")
        sorted_items = sort_items(sorted_items2)
    

    revchan=[None]*len(chanList)
    for idx, name in enumerate(sorted_items):
        revchan[idx] = chanList.index(name)
    revchan.reverse()
    # st.write(revchan)


    doption = st.selectbox("Plot",("Accel", "Vel", "Disp", "FFT for Accel","Floor Spectra"),)
    if doption != None:
        if doption == "Floor Spectra":
            cc1, cc2 = st.tabs(["Damping Ratio", "End Period for Spectra"])
            with cc1:
                xi = st.number_input("Damping Ratio", 0.0, 1.0, 0.05, step=0.01, format="%.2f", key="xi")
            with cc2:
                eT = st.number_input("End Period for Spectra", 0.1, 10.0, 6.0, step=0.1, format="%.2f", key="eT")
        if doption == "FFT for Accel":
            maxFFTylim = st.number_input("Max Hz for FFT", 0.0, 100.0, 10.0, step=0.1, format="%.2f", key="maxFFTylim")
   
        st.write("Plotting " + doption + " for all channels")
        plotAll()


    st.subheader("Floor Drifts")

    resDrift = st.toggle('Resultant Floor Drift Plots', False)
    if resDrift:
        st.write("Select the channels to plot resultant floor drifts")
        selectedFloorsH = st.multiselect("Select two channels in different directions on the same higher floor", sorted_items, default=sorted_items[-1], max_selections=2)
        st.write("#")
        selectedFloorsL = st.multiselect("Select two channels in different directions on the same lower floor", sorted_items, default=sorted_items[-1], max_selections=2) 
        st.write("#") 
        noFlrs = st.number_input("No of floors between channels", min_value=1,step=1, key="numfloorsres")

        if len(selectedFloorsH) == 2 and len(selectedFloorsL) == 2:
            if st.button("Calculate Floor Drifts", key="resDrift"):
                st.write("Calculating Floor Drifts")
                selectedFloorsIndexH = [None]*len(selectedFloorsH)
                selectedFloorsIndexL = [None]*len(selectedFloorsL)
                
                if "360" in selectedFloorsH[0] or " 0 Deg" in selectedFloorsH[0] or "NS" in selectedFloorsH[0]:
                    selectedFloorsH.reverse()
                    
                if "360" in selectedFloorsL[0] or " 0 Deg" in selectedFloorsL[0] or "NS" in selectedFloorsL[0]:
                    selectedFloorsL.reverse()
                
                st.write("Selected Channels on Upper floor: " + selectedFloorsH[0] + " and " + selectedFloorsH[1])
                st.write("Selected Channels on lower Floor: " + selectedFloorsL[0] + " and " + selectedFloorsL[1])   
                
                for idx, name in enumerate(selectedFloorsH):
                    selectedFloorsIndexH[idx] = chanList.index(name)
                for idx, name in enumerate(selectedFloorsL):
                    selectedFloorsIndexL[idx] = chanList.index(name)
                fig6, ax6 = plt.subplots(1,1, sharex='col', sharey='row')
                fig6.set_figheight(width)
                fig6.set_figwidth(width)
                sLoc = int(starttime/dtAccel1[0]); eLoc = int(endtime/dtAccel1[0])
                ax6.plot(displ1[selectedFloorsIndexH[0]][sLoc:eLoc], displ1[selectedFloorsIndexH[1]][sLoc:eLoc],linewidth = 0.5, linestyle="--", label = "Upper Floor")
                ax6.plot(displ1[selectedFloorsIndexL[0]][sLoc:eLoc], displ1[selectedFloorsIndexL[1]][sLoc:eLoc],linewidth = 0.5, linestyle="--", label = "Lower Floor")
                ewdiff = np.subtract(displ1[selectedFloorsIndexH[0]][sLoc:eLoc], displ1[selectedFloorsIndexL[0]][sLoc:eLoc])
                nsdiff = np.subtract(displ1[selectedFloorsIndexH[1]][sLoc:eLoc], displ1[selectedFloorsIndexL[1]][sLoc:eLoc])    
                ax6.plot(ewdiff,nsdiff, color='green',linewidth=2.0, label = "Drift")
                rotmaxLoc = np.argmax(np.sqrt(np.square(ewdiff[:])+np.square(nsdiff[:])))
                resmax = np.sqrt(np.square(ewdiff[rotmaxLoc])+np.square(nsdiff[rotmaxLoc]))
                resAngle = np.arctan2(ewdiff[rotmaxLoc],nsdiff[rotmaxLoc])
                ax6.plot([0,ewdiff[rotmaxLoc]], [0, nsdiff[rotmaxLoc]], color='red',linewidth=2.0 )
                ax6.annotate(str(round(resmax,3)) + "@ " +str(round(resAngle*180/math.pi,2))+ r"$^\circ$", xy=(ewdiff[rotmaxLoc], nsdiff[rotmaxLoc]), xytext=(ewdiff[rotmaxLoc], nsdiff[rotmaxLoc]), fontsize=10, color= 'Blue')
                maxLimit = max(np.max(displ1[selectedFloorsIndexH[0]][sLoc:eLoc]), np.max(displ1[selectedFloorsIndexH[1]][sLoc:eLoc]),np.abs(np.min(displ1[selectedFloorsIndexH[0]][sLoc:eLoc])),np.abs(np.min(displ1[selectedFloorsIndexH[1]][sLoc:eLoc])))/0.95
                maxLimit = max(maxLimit, resmax)
                ax6.set_xlim(-maxLimit, maxLimit)
                ax6.set_ylim(-maxLimit, maxLimit)
                x_left, x_right = ax6.get_xlim()
                y_low, y_high = ax6.get_ylim()
                ax6.set_aspect(abs((x_right-x_left)/(y_low-y_high)))
                xlabel=ax6.get_xticks()
                zind = np.where(xlabel == 0)[0][0]
                for i in range(zind,len(xlabel)):
                    cr = plt.Circle((0, 0), xlabel[i], linestyle="--", color= 'k',linewidth=0.3, fill=False)
                    ax6.add_patch(cr)
                
                ax6.legend()
                ax6.grid()
                st.pyplot(fig6)
                st.write("Max Drift per floor= " + str(round(resmax/noFlrs,3)) + " cm at " + str(round(resAngle*180/math.pi,2)) + r"$^\circ$")


    else:
        selectedFloors = st.multiselect("Select two channels on different floors", sorted_items, default=sorted_items[-1], max_selections=2)
        st.write("#")
        noFlrs = st.number_input("No of floors between channels", min_value=1,step=1, key="numfloors")

        if len(selectedFloors) == 2 and selectedFloors[0] != selectedFloors[1]:
            selectedFloorsIndex = [None]*len(selectedFloors)
            for idx, name in enumerate(selectedFloors):
                selectedFloorsIndex[idx] = chanList.index(name)

            if selectedFloorsIndex[0] > selectedFloorsIndex[1]:
                selectedFloorsIndex.reverse()
                selectedFloors.reverse()
            st.write("Selected Floors: " + selectedFloors[0] + " and " + selectedFloors[1])
            
            
            if st.button("Calculate Floor Drifts"):
                fig5, ax5 = plt.subplots(1,1, sharex='col', sharey='row')
                fig5.set_figheight(height*1.5)
                fig5.set_figwidth(width)
                fig5.suptitle('Floor Drift Calculation', fontsize=11)
                fig5.canvas.manager.set_window_title('Floor Drift Calculation')
                T1 = np.arange(0.0,numofPointsAccel1[0]*dtAccel1[0], dtAccel1[0])
                for i in range(len(selectedFloorsIndex)-1):
                    drift = np.subtract(displ1[selectedFloorsIndex[i+1]], displ1[selectedFloorsIndex[i]])
                    st.write("Drift between " + selectedFloors[i] + " and " + selectedFloors[i+1])
                    st.write("Max Drift = " + str(round(max(abs(drift/noFlrs)),3)) + " cm per floor")
                    ax5.plot(T1, drift, label='Drift')
                    ax5.plot(T1,displ1[selectedFloorsIndex[i]], linewidth = 0.5, linestyle="--",label=selectedFloors[i])
                    ax5.plot(T1,displ1[selectedFloorsIndex[i+1]],linewidth = 0.5, linestyle="--",label=selectedFloors[i+1])
                    ax5.legend()
                    ax5.set_xlim([starttime, endtime])
                    ax5.set_xlabel('Secs')
                    ax5.set_ylabel('Drift (cm)')
                    ax5.grid()
                    st.pyplot(fig5)


    st.subheader("Transfer Function")
    st.write("Select the input and output channels to plot the transfer function")
    

    inChn = st.selectbox("Select Input Chan for transfer function", sorted_items,index=len(sorted_items)-1)
    outChn = st.selectbox("Select output Chan for transfer function", sorted_items)


    if inChn == outChn:
        st.write("Input and Output channels are the same. Please select different channels.")
    else:   
        if st.button("Plot Transfer Function"):
            fig =plottransfer()
            st.pyplot(fig)
    
    st.subheader("Download Ground Motions")
    
    st.download_button("Save Acceleration file", saveFile("Accel"), file_name="accelerations.csv",mime="text/csv",)
    st.download_button("Save Velocity file", saveFile("Vel"), file_name="velocity.csv",mime="text/csv",)
    st.download_button("Save Displacement file", saveFile("Disp"), file_name="displacements.csv",mime="text/csv",)