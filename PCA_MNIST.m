clc; clear; close all; 

load mnist_uint8.mat

train_x = double(train_x);
test_x  = double(test_x);

train_y = double(train_y);
[ ~, trainY ] = max( train_y, [], 2 ) ;
trainY = trainY - 1 ;

test_y  = double(test_y);
[ ~, testY ] = max( test_y, [], 2 ) ;
testY = testY - 1 ;

% Taking 1000 train images
rng('default');
k = randperm( length( train_y ) );
train_x = train_x( k( 1:1000 ), : );
train_y = train_y( k( 1:1000 ), : );
trainY = trainY( k( 1:1000 ) );

% Taking 9000 test images
rng('default');
k = randperm( length( test_y ) );
test_x = test_x( k( 1:9000 ), : );
test_y = test_y( k( 1:9000 ), : );
testY = testY( k( 1:9000 ) );

nTrain = size( train_x, 1 );
nTest = size( test_x, 1 );

%% train PCA
[ tCoeff, tScore, ~, ~, tExp ] = pca( train_x );
tExp = cumsum( tExp );
nEig = find( tExp>99, 1 ); % Taking 99% variance 
nEig = nEig-1;

tCoeff = tCoeff( :, 1:nEig );
tScore = tScore(:, 1:nEig );

%% Classify Digits

mCenTest = bsxfun( @minus, test_x, mean( test_x ) );
proj = mCenTest * tCoeff;

%%
class = zeros( nTest, 1);
tic;
for i = 1:nTest 

    diffMat = bsxfun( @minus, tScore, proj(i,:) );
    [ ~, class( i, 1) ] = min( arrayfun( @(idx) norm( diffMat(idx, :) ), 1:size(diffMat, 1 ) ) );  
    
end
time = toc;

%%
accuracy = zeros(10,1);
totals = zeros(10,1);

confMatrix = zeros(10);
% ntest = length(class);
for i = 1:nTest
    pred = trainY( class(i) );
    label = testY(i);
    
    accuracy( label+1 ) = accuracy(label+1) + (pred==label);
    totals(label+1) = totals(label+1) + 1;
    
    confMatrix(pred+1,label+1) = confMatrix(pred+1,label+1) + 1;
end
accuracy = accuracy ./ totals;
%%
disp(accuracy);
disp(confMatrix);






