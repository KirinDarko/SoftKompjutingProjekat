import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import test
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import winsound

class Kontura:

    koordinate = []

    def __init__(self):
        self.x=0
        self.y=0
        self.broj=[]
        self.dal_Plava=False
        self.dal_Zelena=False
    def _init_(self,x,y,broj,dalzel,dalplav):
        self.x=x
        self.y=y
        self.broj = broj
        self.dal_Zelena = dalzel
        self.dal_Plava = dalplav
    def dodajBroj(self,broj1):
        self.broj.append(broj1)
    def izbaciBroj(self,index):
        del self.broj[index]

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        cm2.imshow('boja',image)
        cv2.waitKey(0)
    else:
        cv2.imshow('crnobelo',image)
        cv2.waitKey(0)
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)
def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)




def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255

def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann

def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def deskew(img,sz1,sz2):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*int((sz1+sz2)/2)*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (sz1, sz2), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def select_roi(image_orig,image_bin):
    img,contours,hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions=[]
    regions_array=[]
    koordinate=[]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        #print(str(x)+','+str(y))
        area = cv2.contourArea(contour)
        #if area > 17 and h > 16 and h < 60 and x>10 and x<630 and y>10 and y<470:
        if area> 17 and h > 16 and h < 60 and x>10 and x<630 and y>10 and y<470:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom

            region = 255-image_bin[y-5:y+h+10,x-5:x+w+10]
            region = cv2.blur(region,(3,3))
            kernel = np.ones((2,2),np.uint8)
            region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
            region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
            region = cv2.dilate(region,kernel,iterations=1)
            region = cv2.erode(region,kernel, iterations=1)
            #region = deskew(region,w,h)
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x-5,y-5),(x+w+10,y+h+10),(0,255,0),2)
            koordinate.append([x+w,y+h])
            #cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    sorted_koords = sorted(koordinate,key=lambda item: item[0])
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions,sorted_koords

def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(26, activation='sigmoid'))
    ann = load_model('treniranmodel.h5')

    return ann


def display_result(outputs, alphabet):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    result = str(alphabet[winner(outputs[0])])
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        
        result += ' '
        result += str(alphabet[winner(output)])
    return result

def get_results(outputs,alphabet):
    results = []
    results.append(alphabet[winner(outputs[0])])
    for idx,output in enumerate(outputs[1:,:]):
        results.append(alphabet[winner(output)])
    return results
def najvise(lista):
    a = np.array(lista)
    counts = np.bincount(a)
    return np.argmax(counts)

file = open('out1.txt','w')
model = load_model('keras_mnist.h5')
video = "C:/Users/darko/Documents/softproj/vezbanje/video-"
for clip in range(0,10):
    video = video+str(clip)+'.avi'

    cap = cv2.VideoCapture(video)

    ret1,frame1 = cap.read()

    cv2.imwrite('kontrola.jpg',frame1)

    mask1 = cv2.inRange(frame1,np.array([130,0,0]),np.array([255,40,40]))
    mask2 = cv2.inRange(frame1,np.array([0,130,0]),np.array([40,255,40]))

    blue1 = cv2.bitwise_and(frame1,frame1,mask=mask1)
    green1 = cv2.bitwise_and(frame1,frame1,mask=mask2)
    gray1 = cv2.cvtColor(blue1,cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray1,50,150,apertureSize = 3)

    gray2 = cv2.cvtColor(green1,cv2.COLOR_BGR2GRAY)
    edges2 = cv2.Canny(gray2,50,150,apertureSize = 3)

    plava = np.array([9999,0,0,9999])
    zelena = np.array([9999,0,0,9999])



    lines1 = cv2.HoughLinesP(edges1,1,np.pi/180,50,None,50,30)
    for x1,y1,x2,y2 in lines1[0]:
        cv2.line(blue1,(x1,y1),(x2,y2),(0,0,255),2)
        if x1<plava[0] and y1>plava[1]:
            plava[0] = x1
            plava[1] = y1
        if x2>plava[2] and y2<plava[3]:
            plava[2]=x2
            plava[3]=y2



    lines2 = cv2.HoughLinesP(edges2,1,np.pi/180,50,None,50,10)
    for x1,y1,x2,y2 in lines2[0]:
        cv2.line(green1,(x1,y1),(x2,y2),(0,0,255),2)

    for x1,y1,x2,y2 in lines2[0]:
        cv2.line(green1,(x1,y1),(x2,y2),(0,0,255),2)
        if x1<zelena[0] and y1>zelena[1]:
            zelena[0] = x1
            zelena[1] = y1
        if x2>zelena[2] and y2<zelena[3]:
            zelena[2]=x2
            zelena[3]=y2

    kplava = (plava[3]-plava[1])/(plava[2]-plava[0])
    kzelena = (zelena[3]-zelena[1])/(zelena[2]-zelena[0])

    nplava = plava[1]-kplava*plava[0]
    nzelena = zelena[1]-kzelena*zelena[0]

    slika = np.zeros((480,640,3), np.uint8)
    cv2.line(slika,(plava[0],plava[1]),(plava[2],plava[3]),(255,0,0),3)
    cv2.line(slika,(zelena[0],zelena[1]),(zelena[2],zelena[3]),(0,255,0),3)
    font = cv2.FONT_HERSHEY_SIMPLEX
        
    cv2.putText(slika,str(plava[0])+', '+str(plava[1]),(plava[0],plava[1]),font,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(slika,str(plava[2])+', '+str(plava[3]),(plava[2],plava[3]),font,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(slika,'Y1=' + str(kplava)+'*X1+ '+str(nplava),(10,plava[1]-50),font,1,(255,255,255),1,cv2.LINE_AA)

    cv2.putText(slika,str(zelena[0])+', '+str(zelena[1]),(zelena[0],zelena[1]),font,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(slika,str(zelena[2])+', '+str(zelena[3]),(zelena[2],zelena[3]),font,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(slika,'Y2=' + str(kzelena)+'*X2+ '+str(nzelena),(10,zelena[1]-50),font,1,(255,255,255),1,cv2.LINE_AA)

    cv2.imwrite('slika.jpg',slika)

    cap = cv2.VideoCapture(video)


    konture=[]
    konacan_rez=0
    broj_iteracija = 0

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        


        #image_color= frame
        if ret is True:
            img = invert(image_bin(image_gray(frame)))
            broj = 0
        else:
            broj = broj +1
            if broj <5:
                continue
            else:
                break
        #img_bin = erode(dilate(img))
        selected_regions, numbers,koords = select_roi(frame.copy(), img)


        #display_image(selected_regions)
        nesto = numbers
        
        cifre = [0,1,2,3,4,5,6,7,8,9]
        outputs = convert_output(cifre)

        inputs = prepare_for_ann(numbers)
        results = model.predict(np.array(inputs, np.float32))
        rezultati = get_results(results,cifre)
        
        if broj_iteracija==0:
            for i,koord in enumerate(koords):
                klasa = Kontura()
                klasa.x=koord[0]
                klasa.y=koord[1]
                klasa.dal_Plava= False
                klasa.dal_Zelena=False
                klasa.dodajBroj(rezultati[i])
                klasa.koordinate.append(koord)
                konture.append(klasa)

        for i,klasa in enumerate(konture):
            if klasa.x>550 or klasa.y>450:
                del konture[i]
            dal = False
            for q in range(-1,7):
                for e in range(-1,7):
                    if [klasa.x+q,klasa.y+e] in koords:
                        dal = True
                        klasa.koordinate[klasa.koordinate.index([klasa.x,klasa.y])]=[klasa.x+q,klasa.y+e]
                        klasa.x = klasa.x+q
                        klasa.y = klasa.y+e
                        break
                if dal is True:
                    break   

        if broj_iteracija>0:
            for i,koord in enumerate(koords):
                klasa = Kontura()
                if koord not in klasa.koordinate:
                    klasa.x=koord[0]
                    klasa.y=koord[1]
                    klasa.dal_Zelena= False
                    klasa.dal_Plava=False
                    klasa.dodajBroj(rezultati[i])
                    klasa.koordinate.append(koord)
                    konture.append(klasa)
                else:
                    del klasa
            for i,klas in enumerate(konture):
                for j,klas1 in enumerate(konture):
                    if i!=j and klas.x==klas1.x and klas.y==klas1.y and klas.broj==klas1.broj:
                        del konture[j]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        
        for i,klasa in enumerate(konture):                             #+0                          +0
            #if klasa.x>=zelena[0]-6 and klasa.x<=zelena[2]+8 and klasa.y>=zelena[3]-6 and klasa.y<=zelena[1]+6 and klasa.y>=((klasa.x)*kzelena+nzelena-3) and klasa.y<=((klasa.x)*kzelena+nzelena) and klasa.dal_Zelena is False:
            if klasa.x>=zelena[0]-6 and klasa.x<=zelena[2]+8 and klasa.y>=zelena[3]-6 and klasa.y<=zelena[1]+6 and klasa.y+15>=((klasa.x-3)*kzelena+nzelena-7) and klasa.y+15<=((klasa.x-3)*kzelena+nzelena) and klasa.dal_Zelena is False:
                konacan_rez = konacan_rez - najvise(klasa.broj)
                #konacan_rez = konacan_rez - klasa.broj[0]
                klasa.dal_Zelena=True

        for i,klasa in enumerate(konture):                           #+15                       +15
            #if klasa.x>=plava[0]-6 and klasa.x<=plava[2]+8 and klasa.y+15>=plava[3]-6 and klasa.y<=plava[1]+6 and klasa.y>=((klasa.x)*kplava+nplava-3) and klasa.y<=((klasa.x)*kplava+nplava) and klasa.dal_Plava is False:
            if klasa.x>=plava[0]-6 and klasa.x<=plava[2]+8 and klasa.y+15>=plava[3]-6 and klasa.y<=plava[1]+6 and klasa.y+15>=((klasa.x-3)*kplava+nplava-7) and klasa.y+15<=((klasa.x-3)*kplava+nplava) and klasa.dal_Plava is False:
                konacan_rez = konacan_rez + najvise(klasa.broj)
                #konacan_rez = konacan_rez + klasa.broj[0]
                klasa.dal_Plava=True
        
        # for index,kont in enumerate(konture):
        #     cv2.putText(selected_regions,str(najvise(kont.broj)),(kont.x,kont.y-10),font,1,(0,255,0),1,cv2.LINE_AA)

        # cv2.putText(selected_regions,str(konacan_rez),(400,300),font,1,(255,0,255),1,cv2.LINE_AA)
        # if (clip==2):
        #     cv2.imshow('frame',selected_regions)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        broj_iteracija = broj_iteracija+1

    print('Rezultat '+str(clip)+': ' + str(konacan_rez))
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    video = "C:/Users/darko/Documents/softproj/vezbanje/video-"
    
    
    if clip==0:
        file.write('RA 67/2014 Darko Kirin\r')
        file.write('file\tsum\r')
    #if clip==2:
    #     file.write('video-'+str(clip)+'.avi	'+str(-3)+'\r')
   # else:
    file.write('video-'+str(clip)+'.avi	'+str(konacan_rez)+'\r')



    if clip==9:
        file.close()
        rezultat = test.testiraj()
        if rezultat>40:
            frequency = 10000  # Set Frequency To 2500 Hertz
            duration = 3000  # Set Duration To 1000 ms == 1 second
        else:
            frequency = 2000  # Set Frequency To 2500 Hertz
            duration = 1000
        winsound.Beep(frequency, duration)
    

