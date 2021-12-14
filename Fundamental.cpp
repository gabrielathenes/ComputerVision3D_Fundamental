// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <math.h>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Returns a vectorof size n of all different integers between 0 and 675
vector<int> random_vector(unsigned int n){
    vector<int> points;
    int r = rand()%675;
    points.push_back(r);
    while (points.size()<n){
        int r = rand()%675;
        if (std::find(points.begin(), points.end(), r)==points.end()) {
            points.push_back(r);
        }
    }
    return points ;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;

    // --------------- TODO ------------

    int best_number_inliers=0;

    //We create a normalisation matrix
    FMatrix<float,3,3> Normalisation = FMatrix<float,3,3>::Zero();
    Normalisation.data()[0]=pow(10,-3);
    Normalisation.data()[4]=pow(10,-3);
    Normalisation.data()[8]=1;

    float Norm=pow(10,-3);
    int flag = 0;

    while (flag==0){
        flag=1 ;

        for (int k=0 ;k<Niter;k++){

            vector<int> random_8 = random_vector(8);
            vector<Match> points_8 ;
            FMatrix<float,9,9> A=FMatrix<float,9,9>::Zero();

            //fill A with 8 normalized random points
            for (int i =0; i<8 ; i++){
                Match match=matches[random_8[i]];
                float x1 = match.x1*Norm;
                float x2 = match.x2*Norm;
                float y1 = match.y1*Norm;
                float y2 = match.y2*Norm;
                A(i,0)=x1*x2;
                A(i,1)=x1*y2;
                A(i,2)=x1;
                A(i,3)=y1*x2;
                A(i,4)=y2*y1;
                A(i,5)=y1;
                A(i,6)=x2;
                A(i,7)=y2;
                A(i,8)=1;
            }
            A.setRow(8, Imagine::FVector<float,9>::Zero());

            // get vector of smallest singular value of A
            FVector<float,9> S;
            FMatrix<float,9,9> U;
            FMatrix<float,9,9> Vt;
            svd(A,U,S,Vt);
            for (int i=0; i<3 ; i++){
                for (int j=0; j<3 ; j++){
                    bestF(i,j)=Vt.getRow(8)[j+3*i];
                }
            }

            // renormalize bestF
            bestF=bestF/norm(bestF);

            // force it to have rank 2
            FVector<float,3> bestFS;
            FMatrix<float,3,3> bestFU;
            FMatrix<float,3,3> bestFVt;
            svd(bestF,bestFU,bestFS,bestFVt);
            bestFS[2]=0;
            bestF=bestFU*(Diagonal(bestFS))*bestFVt;

            // renormalize with the Normalisation matrice
            bestF=Normalisation*bestF*Normalisation;

            int number_inliers = 0;
            vector <int> inliers ;

            //compute number of inliers
            for (int m =0; m<675; m++){
                FVector<float,3> xi;
                FVector<float,3> xiprime;
                xi[0]=matches[m].x1;
                xi[1]=matches[m].y1;
                xi[2]=1;
                xiprime[0]=matches[m].x2;
                xiprime[1]=matches[m].y2;
                xiprime[2]=1;
                float des=abs(xiprime*(transpose(bestF)*xi));
                FVector<float,3> Fp=transpose(bestF)*xi;
                float modul=sqrt(pow(Fp[0],2)+pow(Fp[1],2));
                //la distance = des/modul
                if(des/modul<distMax){
                    inliers.push_back(m);
                    number_inliers++;
                }
            }

            //if number of inliers is better than the best so far, keep model and inliers
            if (number_inliers>best_number_inliers) {
                best_number_inliers=number_inliers;
                float nbr = best_number_inliers ;

                //iter Niter being careful of numerical errors if there is a low number of inliers
                if (best_number_inliers>140){
                    float y= log(BETA)/log(1-pow((nbr/675),8));
                    int x = y;
                    Niter = x;
                }

                bestInliers=inliers ; // keep the inliers as well
                flag=0 ; // since we improved our model we can still do iterations
                break ;
            }
        }


    }

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);


    //Refine bestF over inliers do the same process that we did with matrix A but this time with a bigger matrix B
    const int a = matches.size() ;
    Matrix<float> B(a,9) ;

    for (int i =0; i<a ; i++){
        Match match=matches[i];
        float x1 = match.x1*Norm;
        float x2 = match.x2*Norm;
        float y1 = match.y1*Norm;
        float y2 = match.y2*Norm;
        B(i,0)=x1*x2;
        B(i,1)=x1*y2;
        B(i,2)=x1;
        B(i,3)=y1*x2;
        B(i,4)=y2*y1;
        B(i,5)=y1;
        B(i,6)=x2;
        B(i,7)=y2;
        B(i,8)=1;
    }

    // do SVD with B to get a new best F over inliers
    Vector<float> SB(9);
    Matrix<float> UB(a,9);
    Matrix<float> VtB(9,9);
    svd(B,UB,SB,VtB,true);
    FMatrix<float,3,3> NewBest = FMatrix<float,3,3>::Zero();
    for (int i=0; i<3 ; i++){
        for (int j=0; j<3 ; j++){
            NewBest(i,j)=VtB.getRow(8)[j+3*i];
        }
    }
    NewBest=NewBest/norm(NewBest);

    // force it to have rank 2
    FVector<float,3> NewBestFS;
    FMatrix<float,3,3> NewBestFU;
    FMatrix<float,3,3> NewBestFVt;
    svd(NewBest,NewBestFU,NewBestFS,NewBestFVt);
    NewBestFS[2]=0;
    NewBest=NewBestFU*(Diagonal(NewBestFS))*NewBestFVt;

    // renormalize with the Normalisation matrice
    NewBest=Normalisation*NewBest*Normalisation;

    return NewBest;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    int w = I1.width();

    // we will display the epipoles as well
    FVector <float,3> left_epipole ;
    FVector <float,3> right_epipole ;
    FVector<float,3> FS;
    FMatrix<float,3,3> FU;
    FMatrix<float,3,3> FVt;
    svd(F,FU,FS,FVt);
    right_epipole = FVt.getRow(2)/FVt(2,2) ;
    FVector<float,3> FtransposeS ;
    FMatrix<float,3,3> FtransposeU;
    FMatrix<float,3,3> FtransposeVt;
    svd(transpose(F),FtransposeU,FtransposeS,FtransposeVt);
    left_epipole = FtransposeVt.getRow(2)/FtransposeVt(2,2) ;

    while(true) {
        int x,y;
        FVector<float,3> P;
        if(getMouse(x,y) == 3)
            break;
        else{
            if(x<w){
                P[0]=x;
                P[1]=y;
                P[2]=1;
                Color color(rand()%256,rand()%256,rand()%256);
                drawCircle(x,y,2,color);
                FVector<float,3> right_epiline = transpose(F)*P ;
                IntPoint2 p1 ;
                IntPoint2 p2 ;
                p1.x()=0 ;
                p1.y()=-right_epiline[2]/right_epiline[1];
                p2.x()=w ;
                p2.y()= (-right_epiline[2]-right_epiline[0]*w)/right_epiline[1];
                drawLine(p1.x()+w,p1.y(),p2.x()+w,p2.y(),color);
                drawCircle(right_epipole[0]+w,right_epipole[1],4,RED);
            }
            else{
                P[0]=x-w;
                P[1]=y;
                P[2]=1;
                Color color(rand()%256,rand()%256,rand()%256);
                drawCircle(x,y,2,color);
                FVector<float,3> right_epiline = F*P ;
                IntPoint2 p1 ;
                IntPoint2 p2 ;
                p1.x()=0 ;
                p1.y()=-right_epiline[2]/right_epiline[1];
                p2.x()=w ;
                p2.y()= (-right_epiline[2]-right_epiline[0]*w)/right_epiline[1];
                drawLine(p1.x(),p1.y(),p2.x(),p2.y(),color);
                drawCircle(left_epipole[0],left_epipole[1],4,BLUE);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
