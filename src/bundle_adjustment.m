clear all;
close all;

addpath('../../gtsam_toolbox');

import gtsam.*

tic;
% Options
NUM_FRAMES = 500; % 0 for all
ADD_NOISE = 1;
ITERATIONS = 100;
blenddir = strcat(fileparts(mfilename('fullpath')), '/../blender/');

% Load data
camera_gt = dlmread(strcat(blenddir, 'camera_poses.txt'));
plot(camera_gt(:,2),camera_gt(:,3), 'b.');
hold on;
features_gt = dlmread(strcat(blenddir, 'tracks_dist.txt'));
landmarks_gt = dlmread(strcat(blenddir, 'landmarks_3d.txt'));
landmarks_used = zeros(size(landmarks_gt,1),1);
landmarks_out = zeros(size(landmarks_gt)); % each line is: x,y,z,last_seen_frame_id
varX = zeros(NUM_FRAMES,1);
varY = zeros(NUM_FRAMES,1);
varZ = zeros(NUM_FRAMES,1);


if NUM_FRAMES < 1
    NUM_FRAMES = size(camera_gt, 1);
end
calib = Cal3_S2( ...
    634.8, ... % focal
    634.8, ... % focal
    0, ... % skew
    480,... % center
    270); % center

%% Setup noise
measurementNoiseSigma = 3;
pointNoiseSigma = 0.1;
rotationSigma = 0.2;
positionSigma = 3;
poseNoiseSigmas = [ positionSigma positionSigma positionSigma ...
    rotationSigma rotationSigma rotationSigma]';
posePriorNoise  = noiseModel.Diagonal.Sigmas(poseNoiseSigmas);
pointPriorNoise  = noiseModel.Isotropic.Sigma(3,pointNoiseSigma);
measurementNoise = noiseModel.Isotropic.Sigma(2,measurementNoiseSigma);

%% Add noise to input data
if ADD_NOISE == 1
    disp('Adding noise...')
    
    for i=1:NUM_FRAMES
        
        % 2D features
        f = 1;
        while f < size(features_gt, 2) && features_gt(i,f) > 0
            features_gt(i,f+1:f+2) = features_gt(i,f+1:f+2) + measurementNoiseSigma * randn(1,2);
            f = f + 4;
        end
        
        % Camera Poses
        rot = Rot3.Quaternion( camera_gt(i,8), ...
            camera_gt(i,5), ...
            camera_gt(i,6), ...
            camera_gt(i,7)).xyz;
        
        rot = rot + rotationSigma * randn(3,1);
        temp= Rot3.RzRyRx(rot);
        rotq = temp.quaternion();
        camera_gt(i,8) = rotq(1);
        camera_gt(i,5) = rotq(2);
        camera_gt(i,6) = rotq(3);
        camera_gt(i,7) = rotq(4);
        camera_gt(i,2:4) = camera_gt(i,2:4) + positionSigma * randn(1,3);
    end
    
    % 3D landmarks
    landmarks_gt = landmarks_gt + randn(size(landmarks_gt));
end

graph = NonlinearFactorGraph;
initialEstimate = Values;

% graph.add(PriorFactorPose3(symbol('p',1), get_pose(camera_gt,1), posePriorNoise));

%% Add factors for all measurements
for i=1:NUM_FRAMES
    
    fprintf('Adding frame %d to graph...\n', i);
    
    cam_pose = Pose3(  Rot3.Quaternion(camera_gt(i,8), camera_gt(i,5), camera_gt(i,6), camera_gt(i,7)), ...
        Point3(camera_gt(i,2), camera_gt(i,3), camera_gt(i,4)));
    
    %     << TODO HERE >>
    rdet = det(cam_pose.rotation.matrix);
    if abs(rdet-1) > 0.0001
        fprintf('Correcting R-det: %f \n', rdet);
        [U,S,V]=svd(cam_pose.rotation.matrix);
        % TODO, correct pose (single line missing)
        ortho = U*V';
        cam_pose = Pose3(Rot3(ortho),cam_pose.translation);
    end
    
    initialEstimate.insert(symbol('p',i), cam_pose);
%     graph.add(PriorFactorPose3D(symbol('p',i), cam_pose, posePriorNoise));

    
    f = 1; % column of current feature ID
    x = size(features_gt, 2);
    
    while f < x  && features_gt(i,f) > 0
        
        feature_id = features_gt(i,f);
        
        %         << TODO HERE >>
        % Initialise the point near ground-truth
        if landmarks_used(feature_id,1) < 1
            %             << TODO HERE >>
            landmark_pos = Point3(landmarks_gt(feature_id,1),landmarks_gt(feature_id,2),landmarks_gt(feature_id,3));
            graph.add(PriorFactorPoint3(symbol('f',feature_id), landmark_pos, pointPriorNoise));
            initialEstimate.insert(symbol('f',feature_id),landmark_pos);

            landmarks_used(feature_id,1) = 1;
        end
        
        pt2 = Point2(features_gt(i,f+1),features_gt(i,f+2));
        graph.add(GenericProjectionFactorCal3_S2(pt2,measurementNoise,symbol('p',i),symbol('f',feature_id),calib));
        
        f = f + 4;
    end
    
end


optimizer = LevenbergMarquardtOptimizer(graph,initialEstimate);
initialError = graph.error(initialEstimate);


for i=1:ITERATIONS
    fprintf('Starting iteration %d...\n', i);
    optimizer.iterate();
    result = optimizer.values();
    error = graph.error(result);
    fprintf('Initial error: %f, %d.-iteration error: %f (%3.3f %%)\n', initialError, i, error, 100 * error/ initialError);
    
end

toc;
disp('Retrieving results...');
result = optimizer.values();
fprintf('Initial error: %f\n', graph.error(initialEstimate));
fprintf('Final error: %f\n', graph.error(result));

%% Output noisy input data and optimised result
disp('Exporting results...');
dlmwrite('input_poses.txt',camera_gt(1:NUM_FRAMES,:),'delimiter','\t','precision',6);
output_poses = zeros(NUM_FRAMES,size(camera_gt,2));
for p = 1:NUM_FRAMES
    pose = result.atPose3(symbol('p', p));
    pos = pose.translation();
    quat = pose.rotation().quaternion();
    output_poses(p,:) = [camera_gt(p,1) pos.x pos.y pos.z quat(4) quat(2) quat(3) quat(1)];
end

i=NUM_FRAMES;
plot(camera_gt(1:i,2),camera_gt(1:i,3), 'r*');
hold on
plot(output_poses(1:i,2),output_poses(1:i,3), 'g+');


dlmwrite('output_poses.txt',output_poses,'delimiter','\t','precision',6);

% Code to plot the 3D map of the corrected landmarks
for l=1:length(landmarks_used)
    if landmarks_used(l) == 1
        mark = result.atPoint3(symbol('f', l));
        landmarks_out(l,:) = [mark.x mark.y mark.z];
    end
end

% Use pcshow to show the point cloud of corrected landmarks
figure
pcshow(pointCloud(landmarks_out));
figure;


% Code to extract the diaginal elements of covariance of pose of camera
marginals = Marginals(graph, result);

for i=1:NUM_FRAMES
    pose_i = result.atPose3(symbol('p', i));
    cov=marginals.marginalCovariance(symbol('p', i));
    diagCov = diag(cov);
    varX(i,1) = diagCov(4);
    varY(i,1) = diagCov(5);
    varZ(i,1) = diagCov(6);
end

% Code to plot the variance of X, Y  and X component of translation of
% camera pose.
figure
plot(varX)
xlabel('Frames');
ylabel('Variance of X component of translation of Pose')
figure
plot(varY)
xlabel('Frames');
ylabel('Variance of Y component of translation of Pose')
figure
plot(varZ)
xlabel('Frames');
ylabel('Variance of Z component of translation of Pose')
disp('done');

function p = get_pose(matrix, index)
import gtsam.*;
p = Pose3(Rot3.Quaternion(matrix(index,8), matrix(index,5), matrix(index,6), matrix(index,7)), ...
    Point3(matrix(index,2), matrix(index,3), matrix(index,4)));
end