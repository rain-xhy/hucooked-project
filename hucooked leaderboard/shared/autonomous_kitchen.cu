// Batched score dynamics for the default three-player continuous SharedKitchen.
// Chef order is canonical spawn order. Provenance and diagnostics stay in the native referee.
#include <cuda_runtime.h>
#include <stdint.h>

enum Field { Pos, Face=3, Hand=6, Pot=9, Cook, Cut, Side, Plate, Main, Buffer,
             Loaded, Served, Tick, Fields };
enum Item { Empty, RawMain, RawSide, Chopped, Dish, Meal };
enum Op { NoOp, GetMain, GetSide, GetPlate, Return, Load, PlaceCut, Chop, TakeCut,
          PutBuffer, TakeBuffer, PutSide, PutPlate, Assemble, TakeMeal, Deliver };
__device__ const char layout[]="XXXXMXXX     LX  G  KS  A  XX  B  XD     XXXXXXXX";
__device__ const int delta[4]={-7,7,1,-1};

__device__ int transition(int* saved, const int64_t* a, int* success) {
    int s[Fields];
    for(int i=0;i<Fields;i++) s[i]=saved[i];

    int op[3]={}, resource[3]={}, desired[3], empty_at_pot=0;
    int old_pot=s[Pot], old_cook=s[Cook];
    for(int i=0;i<3;i++) {
        int pos=s[Pos+i], face=s[Face+i], hand=s[Hand+i];
        desired[i]=pos;
        if(a[i]<4) {
            int dest=pos+delta[a[i]];
            if(layout[dest]==' ') desired[i]=dest;
            s[Face+i]=a[i];
        }
        if(a[i]!=5) continue;
        char target=layout[pos+delta[face]];
        if(target=='M') op[i]=hand==Empty?GetMain:(hand==RawMain?Return:NoOp);
        if(target=='L') op[i]=hand==Empty?GetSide:((hand==RawSide||hand==Chopped)?Return:NoOp);
        if(target=='D') op[i]=hand==Empty?GetPlate:(hand==Dish?Return:NoOp);
        if(target=='G' && (pos==16||pos==18)) {
            if(hand==Empty) empty_at_pot++;
            if(s[Pot]<0 && hand==RawMain && s[Loaded]<s[Served]+2) {
                op[i]=Load; resource[i]=1;
            }
        }
        if(target=='K') {
            if(!s[Cut] && hand==RawSide) op[i]=PlaceCut;
            if(s[Cut] && hand==Empty) op[i]=s[Cut]==3?TakeCut:Chop;
            if(op[i]) resource[i]=2;
        }
        if(target=='B') {
            if(hand && !s[Buffer]) op[i]=PutBuffer;
            if(!hand && s[Buffer]) op[i]=TakeBuffer;
            if(op[i]) resource[i]=4;
        }
        if(target=='A' && (pos==23||pos==25)) {
            if(hand==Chopped&&!s[Side]) {op[i]=PutSide;resource[i]=8;}
            else if((hand==Dish||hand==Meal)&&!s[Plate]) {op[i]=PutPlate;resource[i]=16;}
            else if(!hand&&s[Plate]==Meal) {op[i]=TakeMeal;resource[i]=16;}
            else if(!hand&&s[Plate]&&s[Side]&&s[Main]>=0) {op[i]=Assemble;resource[i]=8|16|32;}
        }
        if(target=='S' && hand==Meal) op[i]=Deliver;
    }
    bool unload=s[Pot]>=0 && !s[Cook] && s[Main]<0 &&
                ((s[Pot]%2==0 && empty_at_pot>0)||(s[Pot]%2==1 && empty_at_pot==2));
    if(unload) {
        for(int i=0;i<3;i++)
            if(a[i]==5 && s[Hand+i]==Empty && (s[Pos+i]==16||s[Pos+i]==18) &&
               layout[s[Pos+i]+delta[s[Face+i]]]=='G') success[i]=16;
        s[Main]=s[Pot];s[Pot]=-1;
    }
    int delivered=0;
    for(int i=0;i<3;i++) {
        if(resource[i]&(resource[(i+1)%3]|resource[(i+2)%3])) continue;
        int& hand=s[Hand+i];
        if(op[i]) success[i]=op[i];
        switch(op[i]) {
        case GetMain: hand=RawMain; break;
        case GetSide: hand=RawSide; break;
        case GetPlate: hand=Dish; break;
        case Return: hand=Empty; break;
        case Load: s[Pot]=s[Loaded]++;s[Cook]=4;hand=Empty;break;
        case PlaceCut: s[Cut]=1;hand=Empty;break;
        case Chop: s[Cut]++;break;
        case TakeCut: s[Cut]=0;hand=Chopped;break;
        case PutBuffer: s[Buffer]=hand;hand=Empty;break;
        case TakeBuffer: hand=s[Buffer];s[Buffer]=0;break;
        case PutSide: s[Side]=1;hand=Empty;break;
        case PutPlate: s[Plate]=hand;hand=Empty;break;
        case Assemble: s[Plate]=Meal;s[Main]=-1;s[Side]=0;break;
        case TakeMeal: hand=Meal;s[Plate]=Empty;break;
        case Deliver: s[Served]++;delivered++;hand=Empty;break;
        }
    }
    bool blocked[3]={};
    for(int i=0;i<3;i++) for(int j=i+1;j<3;j++)
        if(desired[i]==desired[j] || (desired[i]==s[Pos+j] && desired[j]==s[Pos+i]))
            blocked[i]=blocked[j]=true;
    for(int pass=0;pass<3;pass++) for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        if(i!=j && desired[i]==s[Pos+j] && (blocked[j]||desired[j]==s[Pos+j])) blocked[i]=true;
    for(int i=0;i<3;i++) if(!blocked[i]) s[Pos+i]=desired[i];
    if(old_pot>=0 && old_cook>0 && old_pot==s[Pot]) s[Cook]--;
    s[Tick]++;
    for(int i=0;i<Fields;i++) saved[i]=s[i];
    return delivered;
}
__global__ void advance(int* batch, const int64_t* actions, float* rewards, int n) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    int success[3]={};
    rewards[row]=20*transition(batch+row*Fields,actions+row*3,success);
}
extern "C" void shared_step(int* s,const int64_t* a,float* r,int n,cudaStream_t stream) {
    advance<<<(n+127)/128,128,0,stream>>>(s,a,r,n);
}

// A sampled manager maps each chef's progress counter to a learned interaction option.
// Timing is event-driven: attempting interaction before others arrive is allowed.
__global__ void play_options(const short* programs,const int* lengths,const unsigned char* nav,
                             int stride,int n,float* result,int* trace) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    int s[Fields]={23,18,19,0,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0};
    int pc[3]={};
    int weighted=0;
    for(int tick=0;tick<300;tick++) {
        int64_t a[3];
        int expected[3], success[3]={};
        for(int i=0;i<3;i++) {
            int token=programs[(row*3+i)*stride+pc[i]];
            expected[i]=token%32;
            int goal=token/32;
            a[i]=lengths[row*3+i]?nav[(goal*49+s[Pos+i])*4+s[Face+i]]:4;
            if(trace) trace[(row*300+tick)*3+i]=a[i];
        }
        int meals=transition(s,a,success);
        weighted+=meals*(300-tick);
        for(int i=0;i<3;i++)
            if(success[i] && success[i]==expected[i] && ++pc[i]==lengths[row*3+i]) pc[i]=0;
    }
    result[2*row]=s[Served]*20;
    result[2*row+1]=weighted;
}
extern "C" void option_returns(const short* p,const int* lengths,const unsigned char* nav,
            int stride,int n,float* result,int* trace,cudaStream_t stream) {
    play_options<<<(n+127)/128,128,0,stream>>>(p,lengths,nav,stride,n,result,trace);
}
