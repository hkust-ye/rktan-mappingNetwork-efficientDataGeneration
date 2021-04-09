import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from neuralnet import network
import tensorflow as tf

np.random.seed(1)

def KNE(flag,poi):
    KE0=KE_0(flag,1,poi,1,-1,-1,1,-1,1,1,-1,1)
    NE0=NE_0()
    NET0=NET_0()
    KE1=np.zeros((16,16))
    NE1=np.zeros((3,16))
    NET1=np.zeros((16,2))
    d=np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]])
    dd=np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]])
    fg1=np.array([0,1,2,3,4,5,6,7])
    fg2=np.array([6,7,0,5,1,4,3,2])
    for i in range(8):
        for j in range(8):
            t1=fg1[i]
            t2=fg1[j]
            t3=fg2[i]
            t4=fg2[j]
            KE1[d[t1,:].reshape(-1,1),d[t2,:]]=KE0[dd[t3,:].reshape(-1,1),dd[t4,:]]
    for i in range(8):
            t1=fg1[i]
            t2=fg2[i]
            NE1[:,2*t1:2*t1+2]=NE0[:,2*t2:2*t2+2]
    for i in range(8):
            t1=fg1[i]
            t2=fg2[i]
            NET1[2*t1:2*t1+2,:]=NET0[2*t2:2*t2+2,:]
    return KE1,NE1,NET1

def KE_0(flag,E,v,h,x1,y1,x3,y3,x5,y5,x7,y7):
    Ke=np.zeros((16,16));
    if flag==1:
       D=E/(1-v*v)*np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]]); # elastic matrix
    else:
       D=E/((1+v)*(1-2*v))*np.array([[1-v,v,0],[v,1-v,0],[0,0,(1-2*v)/2]])
    w1=5/9
    w2=8/9
    w3=5/9
    w=np.array([w1,w2,w3])
    r=15**0.5/5
    x=np.array([-r,0,r])
    for i in range(3):
        for j in range(3):
            B=eleB(x1,y1,x3,y3,x5,y5,x7,y7,x[i],x[j])
            J=Jacobi(x1,y1,x3,y3,x5,y5,x7,y7,x[i],x[j])
            Ke=Ke+w[i]*w[j]*np.dot(np.dot(B.T,D),B)*np.linalg.det(J)*h # here det(J) is for the integration area transformation.
    return Ke


def eleB(x1,y1,x3,y3,x5,y5,x7,y7,s,t):
    N_s,N_t=DHS(s,t);
    J=Jacobi(x1,y1,x3,y3,x5,y5,x7,y7,s,t);
    B=np.zeros((3,16));
    for i in range(8):
        B1i=J[1,1]*N_s[i]-J[0,1]*N_t[i]
        B2i=-J[1,0]*N_s[i]+J[0,0]*N_t[i]
        B[:,2*i:2*(i+1)]=np.array([[B1i,0],[0,B2i],[B2i,B1i]])
    B=B/np.linalg.det(J);
    return B

    
def Jacobi(x1,y1,x3,y3,x5,y5,x7,y7,s,t):    
    x2=(x1+x3)/2; y2=(y1+y3)/2;
    x4=(x3+x5)/2;y4=(y3+y5)/2;
    x6=(x5+x7)/2; y6=(y5+y7)/2;
    x8=(x7+x1)/2;y8=(y7+y1)/2;
    x=np.array([x1,x2,x3,x4,x5,x6,x7,x8])
    y=np.array([y1,y2,y3,y4,y5,y6,y7,y8])
    N_s,N_t=DHS(s,t)
    x_s=0
    y_s=0
    x_t=0 
    y_t=0
    for i in range(8):
        x_s=x_s+N_s[i]*x[i]
        y_s=y_s+N_s[i]*y[i]
        x_t=x_t+N_t[i]*x[i]
        y_t=y_t+N_t[i]*y[i]
    J=np.array([[x_s,y_s],[x_t,y_t]])
    return J


def DHS(s,t):
    N_s=np.zeros(8)
    N_t=np.zeros(8)
    N_s[0]=-1/4*(1-t)*(-s-t-1)-1/4*(1-s)*(1-t)
    N_s[2]=1/4*(1-t)*(s-t-1)+1/4*(1+s)*(1-t)
    N_s[4]=1/4*(1+t)*(s+t-1)+1/4*(1+s)*(1+t)
    N_s[6]=-1/4*(1+t)*(-s+t-1)-1/4*(1-s)*(1+t)
    N_s[1]=1/2*(1-s)*(1-t)-1/2*(1+s)*(1-t)
    N_s[3]=1/2*(1+t)*(1-t)
    N_s[5]=1/2*(1-s)*(1+t)-1/2*(1+s)*(1+t)
    N_s[7]=-1/2*(1+t)*(1-t)
    N_t[0]=-1/4*(1-s)*(-s-t-1)-1/4*(1-s)*(1-t)
    N_t[2]=-1/4*(1+s)*(s-t-1)-1/4*(1+s)*(1-t)
    N_t[4]=1/4*(1+s)*(s+t-1)+1/4*(1+s)*(1+t)
    N_t[6]=1/4*(1-s)*(-s+t-1)+1/4*(1-s)*(1+t)
    N_t[1]=-1/2*(1+s)*(1-s)
    N_t[3]=1/2*(1+s)*(1-t)-1/2*(1+s)*(1+t)
    N_t[5]=1/2*(1+s)*(1-s)
    N_t[7]=1/2*(1-s)*(1-t)-1/2*(1-s)*(1+t)
    return N_s,N_t


def NE_0():
    BN=np.zeros((3,16))
    w1=5/9; w2=8/9; w3=5/9;
    w=np.array([w1,w2,w3])
    r=15**0.5/5
    x=np.array([-r,0,r])
    for i in range(3):
        for j in range(3):
            a=NE_F(x[i],x[j]);
            BN[:,:]=BN[:,:]+w[i]*w[j]*a;
    BN=BN*1/2;
    return BN


def NE_F(x,y):
    PN=np.zeros((8,2))
    #N(0,i,j)=(1-x)*(1-y)*(-x-y-1)/4;
    PN[0,0]=(2*x+y)*(1-y)/4;
    PN[0,1]=(x+2*y)*(1-x)/4;
   
    #N(1,i,j)=(1-x^2)*(1-y)/2;
    PN[1,0]=-2*x*(1-y)/2;
    PN[1,1]=-(1-x**2)/2;
       
    #N(2,i,j)=(1+x)*(1-y)*(x-y-1)/4;
    PN[2,0]=(2*x-y)*(1-y)/4;
    PN[2,1]=(-x+2*y)*(1+x)/4;
      
    #N(3,i,j)=(1+x)*(1-y^2)/2;
    PN[3,0]=(1-y**2)/2;
    PN[3,1]=(1+x)*(-2*y)/2;
       
    #N(4,i,j)=(1+x)*(1+y)*(x+y-1)/4;
    PN[4,0]=(2*x+y)*(1+y)/4;
    PN[4,1]=(1+x)*(x+2*y)/4;
       
    #N(5,i,j)=(1-x^2)*(1+y)/2;
    PN[5,0]=-2*x*(1+y)/2;
    PN[5,1]=(1-x**2)/2;
            
    #N(6,i,j)=(1-x)*(1+y)*(-x+y-1)/4;
    PN[6,0]=(2*x-y)*(1+y)/4;
    PN[6,1]=(1-x)*(-x+2*y)/4;

    #N(7,i,j)=(1-x)*(1-y^2)/2;
    PN[7,0]=-(1-y**2)/2;
    PN[7,1]=(1-x)*(-2*y)/2;

    B=np.zeros((3,16))
    for i in range(8):
        B[:,i*2:(i+1)*2]=np.array([[PN[i,0],0],[0,PN[i,1]],[PN[i,1],PN[i,0]]])
    return B


def NET_0():
    NET=np.zeros((16,2));
    w1=5/9; w2=8/9; w3=5/9;
    w=np.array([w1,w2,w3])
    r=15**0.5/5;
    x=np.array([-r,0,r])
    for i in range(3):
        for j in range(3):
            a=NET_F0(x[i],x[j])
            NET[:,:]=NET[:,:]+w[i]*w[j]*a;
    NET=NET/4
    return NET

def NET_F0(x,y):
    N=np.zeros(8);
    N[0]=(1-x)*(1-y)*(-x-y-1)/4;
    N[1]=(1-x**2)*(1-y)/2;
    N[2]=(1+x)*(1-y)*(x-y-1)/4;
    N[3]=(1+x)*(1-y**2)/2;
    N[4]=(1+x)*(1+y)*(x+y-1)/4;
    N[5]=(1-x**2)*(1+y)/2;
    N[6]=(1-x)*(1+y)*(-x+y-1)/4;
    N[7]=(1-x)*(1-y**2)/2;

    NET_F=np.zeros((16,2))
    NET_F[0::2,0]=N[:]
    NET_F[1::2,1]= NET_F[0::2,0]
    return NET_F

def stress_and_compliance(x0,fx0,fy0):
    p=4 #p-norm stress measurement
    ps=3#p in SIMP scheme
    nely=np.size(x0,0)
    nelx=np.size(x0,1)
    dx=2.5e-3*0.5
    dy=2.5e-3*0.5
    nNode=(nelx*(3*nely+2))+2*nely+1
    xPhys=np.array(x0)
    E0=69e9
    Emin=1e-8*E0
    poi=0.3
    CP=E0/(1-poi*poi)*np.array([[1,poi,0],[poi,1,0],[0,0,(1-poi)/2]])
    flag=1
    KE,NE,NET=KNE(flag,poi) 
    eleN=np.matlib.repmat(np.array(range(0,2*nely,2)),1,nelx)+np.kron(np.array(range(0,nelx)),(3*nely+2)*np.ones(nely))# node arrangement
    eleNode=np.matlib.repmat(eleN.T,1,8)+np.matlib.repmat(np.concatenate((np.matlib.repmat(np.array([[0,1,2]]),nely,1),np.array(range(2*nely+1,(nely+1),-1)).reshape(nely,1),np.array(range(2*nely+2,nely+2,-1)).reshape(nely,1),np.matlib.repmat(np.array([[3*nely+2,3*nely+3,3*nely+4]]),nely,1)),axis=1),nelx,1)  # node arrangement(each point contain 2 nodes, to mean 'x' and 'y' direction)
    edofMat =(np.matlib.repmat(np.array([[0,1]]),nelx*nely,8)+np.kron(eleNode,np.array([[2,2]]))).astype(int)       
    iK = np.kron(edofMat,np.ones((16,1))).reshape(16*16*nelx*nely,1)
    jK = np.kron(edofMat,np.ones((1,16))).reshape(16*16*nelx*nely,1)

    flag=np.dot(KE.reshape(16*16,1),(Emin+(xPhys.T).reshape(1,nelx*nely)*(E0-Emin)))
    sK= flag.T.reshape(16*16*nelx*nely,1)
    K = sparse.csr_matrix((sK.flatten(),(iK.flatten(),jK.flatten())),shape=(nNode*2,nNode*2)) 
    K = (K+K.T)/2

    degL=np.array(range(2*(2*nely+1)))
    degO=np.array(range(2*(2*nely+1),2*nNode))
    UL=np.zeros((np.size(degL),1))
    UO=np.zeros((np.size(degO),1))

    TU=np.zeros((2*nNode,1))
    
    TF_p=np.zeros((2*nNode,1)) # simplly verified
    F_eNode=(1.0e8)*(dx*dy)*np.dot(NET,np.concatenate(((fx0.T).reshape(1,nelx*nely),(fy0.T).reshape(1,nelx*nely)),axis=0))
    iKNET = edofMat.reshape(16*nelx*nely,1)
    TF_p=sparse.csr_matrix((F_eNode.T.flatten(),(iKNET.flatten(),np.zeros(16*nelx*nely))),shape=(nNode*2,1)) # suppose that it is correct

    FO=TF_p[degO]
    
    for test_case in range(1):
        TK=K
        K21=TK[degO.reshape(-1,1),degL]
        K22=TK[degO.reshape(-1,1),degO]
       
        NKN=sparse.csr_matrix(K22)
        NFN=sparse.csr_matrix((FO-K21*UL).reshape(-1,1))
        NU=spsolve(NKN,NFN)         
        UO=NU.reshape(-1,1)
        TU[0:np.size(degL)]=UL
        TU[np.size(degL):nNode*2]=UO
    Xigem_e=np.zeros((nelx*nely,3))
    x_temp=(xPhys.T).reshape(1,nelx*nely)
    compliance=0
    compliance_e=np.zeros(nelx*nely)
    for num in range(nelx*nely):
        Ue=TU[edofMat[num,:],0].reshape(-1,1)
        Xigem_e[num,:]=1/(dx)*np.dot(np.dot(CP,NE),Ue).T*(Emin+x_temp[0,num]*(E0-Emin))/E0
        compliance=compliance+(Emin+x_temp[0,num]*(E0-Emin))*np.dot(np.dot(Ue.T,KE),Ue)
        compliance_e[num]=(Emin+x_temp[0,num]*(E0-Emin))*np.dot(np.dot(Ue.T,KE),Ue)
    print('cmp=',compliance)
    return Xigem_e,compliance,compliance_e

def filtering(f,rmin,nelx,nely):
    iH = np.ones((int(nelx*nely*(2*np.fix(rmin)+1)**2),1))
    #print(iH.shape)
    jH = np.ones((iH.shape))
    sH = np.zeros((iH.shape))
    k = 0
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1*nely+j1;
            for i2 in range(int(max((i1-np.fix(rmin)),0)),int(min(i1+np.fix(rmin)+1,nelx))):
                for j2 in range(int(max((j1-np.fix(rmin)),0)),int(min(j1+np.fix(rmin)+1,nely))):
                    e2 = i2*nely+j2;
                    iH[k,:] = e1
                    jH[k,:] = e2
                    sH[k,:] = max(0,rmin-((i1-i2)**2+(j1-j2)**2)**0.5)
                    k = k+1
    H = sparse.csr_matrix((sH.flatten(),(iH.flatten().astype(int),jH.flatten().astype(int)))).todense()
   
    Hs = np.sum(H,axis=1)
    filtered_value=np.dot(H,f)/Hs
    return np.asarray(filtered_value)


def comp_pred(xphy,fy0,nelx,nely):
    x_data1 = xphy.reshape(1,nelx,nely,1)
    x_data2 = fy0.reshape(1,nelx,nely,1)
    x_data = np.concatenate((x_data1,x_data2),axis=3)
    predicted_comp = sess.run(prediction,feed_dict={gen_input:x_data})
    comp_output = np.maximum(predicted_comp*1e-4,0)

    print('cmp=',np.sum(comp_output))

    return np.sum(comp_output),comp_output.reshape(1,nelx*nely)

def optimization(nelx,nely,fx0,fy0,Vol0,case_num):
    iter_num=100
    p=4
    error1=1
    error2=1
    eleNum=nelx*nely
    Cmp=np.zeros((iter_num,1))
    Pd_filtered=np.zeros((100,eleNum))
    Rf_s=6
    Rf_d=8
    
    x_ini=np.ones((eleNum,1))   
    x_temp=filtering(x_ini,Rf_d,nelx,nely) 
    Cer=0.025
    Vol=np.sum(x_temp)
    VolC=np.sum(x_temp)   #current desired volume
    VolN=int(np.round(Vol0*eleNum))  # final required volume
    Car_max=int(np.round(0.02*eleNum))


    xflag=np.argsort(x_temp[:,0])
    xphy=1.0*(x_temp>=x_temp[xflag[int(nelx*nely-VolC)]]).reshape(nelx*nely,1)
    if fem0nn1 == 0:
        Xigem_e,compliance,compliance_e=stress_and_compliance(xphy.reshape(nelx,nely).T,fx0,fy0)
    if fem0nn1 == 1:
        compliance, compliance_e = comp_pred(xphy.reshape(nelx,nely),fy0.T,nelx,nely)

    itr=0
    Cmp[itr,0]=compliance

    fi0=open(savepath+'/dens'+str(case_num)+'_'+str(Vol0)+'.csv','ab')
    np.savetxt(fi0,xphy.reshape(1,nelx*nely),fmt='%1d',delimiter=',')
    fi0.close()
    fi2=open(savepath+'/fy'+str(case_num)+'_'+str(Vol0)+'.csv','ab')
    np.savetxt(fi2,fy0.reshape(1,nelx*nely),fmt='%e',delimiter=',')
    fi2.close()

    fi3=open(savepath+'/comp'+str(case_num)+'_'+str(Vol0)+'.csv','ab')
    np.savetxt(fi3,compliance_e.reshape(1,nelx*nely),fmt='%e',delimiter=',')
    fi3.close()
    for itr in range(iter_num-1):
          print('itr=',itr,' of:',case_num,'_',Vol0) 
          Pd_filtered[itr,:]=-filtering(xphy*((compliance_e/compliance).reshape(-1,1)),Rf_s,nelx,nely).reshape(-1,)
          if itr==0:
             BT=-np.array(Pd_filtered[itr,:])
          if itr==1:
             BT=-(Pd_filtered[itr,:]+Pd_filtered[itr-1,:])/2
          if itr>=2:
             BT=-(Pd_filtered[itr,:]+Pd_filtered[itr-1,:]+Pd_filtered[itr-2,:])/3
          BT_ranking=np.argsort(BT)

          VolC=int(max(np.round(Vol0*eleNum),np.round((1-Cer)*VolC)))  
          flag1=0
          flag2=eleNum-1
          flag=int((flag1+flag2)/2)
          BT0=1.0/2*(BT[BT_ranking[flag]]+BT[BT_ranking[flag+1]])
          x_temp=np.array(-1.0*(BT<=BT0).reshape(-1,1)*(xphy==1)+1.0*(BT>BT0).reshape(-1,1)*(xphy==0)+xphy)
          Volt=np.sum(x_temp)
          while Volt!=VolC and flag2-flag1>1:
                if Volt<VolC:
                    flag1=flag1
                    flag2=flag
                else:
                    flag1=flag
                    flag2=flag2
                flag=int((flag1+flag2)/2)
                
                BT0=1.0/2*(BT[BT_ranking[flag]]+BT[BT_ranking[flag+1]])
                x_temp=np.array(-1.0*(BT<=BT0).reshape(-1,1)*(xphy==1)+1.0*(BT>BT0).reshape(-1,1)*(xphy==0)+xphy)
                Volt=np.sum(x_temp)
          
          BTA=BT0
          Car=np.sum((BT>BT0).reshape(-1,1)*(xphy==0))
          flagA=flag
          while Car>Car_max and flagA<nelx*nely-1:
                flagA=flagA+1
                BTA=1.0/2*(BT[BT_ranking[flagA]]+BT[BT_ranking[flagA-1]])
                Car=np.sum((BT>BTA).reshape(-1,1)*(xphy==0))
          BTD=BT0
          x_temp=np.array(-1.0*(BT<=BTD).reshape(-1,1)*(xphy==1)+1.0*(BT>BTA).reshape(-1,1)*(xphy==0)+xphy)
          Volt=np.sum(x_temp)
          flagD=flag
          while Volt<VolC and flagD>1:
                flagD=flagD-1
                BTD=1.0/2*(BT[BT_ranking[flagD]]+BT[BT_ranking[flagD+1]])
                x_temp=np.array(-1.0*(BT<=BTD).reshape(-1,1)*(xphy==1)+1.0*(BT>BTA).reshape(-1,1)*(xphy==0)+xphy)
                Volt=np.sum(x_temp)


          x_temp1=filtering(x_temp,Rf_d,nelx,nely)
          xflag=np.argsort(x_temp1[:,0])
          xphy=1.0*(x_temp1>=x_temp1[xflag[int(nelx*nely-VolC)]]).reshape(nelx*nely,1) 
          if fem0nn1 == 0:
            Xigem_e,compliance,compliance_e=stress_and_compliance(xphy.reshape(nelx,nely).T,fx0,fy0)
            print('fem calc--')
          if fem0nn1 == 1:
            compliance, compliance_e = comp_pred(xphy.reshape(nelx,nely),fy0.T,nelx,nely)
            print('network calc--')


          fi0=open(savepath+'/dens'+str(case_num)+'_'+str(Vol0)+'.csv','ab')
          np.savetxt(fi0,xphy.reshape(1,nelx*nely),fmt='%1d',delimiter=',')
          fi0.close()
          fi2=open(savepath+'/fy'+str(case_num)+'_'+str(Vol0)+'.csv','ab')
          np.savetxt(fi2,fy0.reshape(1,nelx*nely),fmt='%e',delimiter=',')
          fi2.close()

          fi3=open(savepath+'/comp'+str(case_num)+'_'+str(Vol0)+'.csv','ab')
          np.savetxt(fi3,compliance_e.reshape(1,nelx*nely),fmt='%e',delimiter=',')
          fi3.close()
          
          itr=itr+1
         
          Cmp[itr,0]=compliance
          if itr>10: 
             c1=np.sum(Cmp[itr-9:itr-4,:])
             c2=np.sum(Cmp[itr-4:itr+1,:])
             error2=np.abs(c1-c2)/max(1e-5,np.abs(c2))
             
          Vol=np.sum(xphy)   
          if (abs(1-Vol/VolN)<0.005 and error2<1e-3 and itr>25):
               break
    return xphy,compliance_e



# main:

# vol=np.array([0.3,0.4,0.5,0.6,0.7])
vol=np.array([0.5])
savepath = 'opt_result'
nelx=128
nely=128
fem0nn1 = 1
x0=np.ones((nely,nelx))
fx0=np.zeros((nely,nelx))
fy0=np.zeros((nely,nelx))

wd=8
gap=8

gen_input = tf.placeholder(tf.float32, [None, nelx, nely, 2])
prediction = network(gen_input)
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    savefileid = 'save_prop/model100'
    saver.restore(sess, savefileid)
    for i in range(6):
        for vo in range(vol.size):
            flagy=(gap*i)
            fx0=np.zeros((nely,nelx))
            fy0=np.zeros((nely,nelx))

            print('case=',i)
            print('vol=',vol[vo])
            for j in range(wd):
                fy0[flagy+j,nelx-1]=1
                fy0[flagy+j,nelx-2]=1
                fy0[flagy+j,nelx-3]=1
                fy0[flagy+j,nelx-4]=1
                fy0[flagy+j,nelx-5]=1
                fy0[flagy+j,nelx-6]=1
                fy0[flagy+j,nelx-7]=1
                fy0[flagy+j,nelx-8]=1

            d_m,compliance_m=optimization(nelx,nely,fx0,fy0,vol[vo],(i))
            

            fi1=open(savepath+'/dens_final.csv','ab')
            np.savetxt(fi1,d_m.reshape(1,nelx*nely),fmt='%1d',delimiter=',')
            fi1.close()

            fi3=open(savepath+'/comp_final.csv','ab')
            np.savetxt(fi3,compliance_m.reshape(1,nelx*nely),fmt='%e',delimiter=',')
            fi3.close()
            print('case completed:',i,' of ',vol[vo])
