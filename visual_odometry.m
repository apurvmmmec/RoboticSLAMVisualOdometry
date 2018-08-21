close all
clear all

addpath(genpath('../../gtsam_toolbox'))

import gtsam.*

format long;

%% Data Options
NUM_FRAMES = 0; % 0 for all
NUM_INITIALISE =10 %TODO find a nice values
WINDOW_SIZE = 5%TODO find a nice value
MAX_ITERATIONS = 30;
LOOP_THRESHOLD = 1000;

ADD_NOISE = true; % Try false for debugging (only)
SAVE_FIGURES = true; % Save plots to hard-drive?
PLOT_LANDMARKS = true;
PLOT_FEATURES = true;

outdir = strcat(fileparts(mfilename('fullpath')), '/output/');
outdir_ate = strcat(fileparts(mfilename('fullpath')), '/output_ate/');
outdir_gt = strcat(fileparts(mfilename('fullpath')), '/groundTruth_window/');

blenddir = strcat(fileparts(mfilename('fullpath')), '/../blender/');
mkdir(outdir);

%% Load data
global features_gt;
global camera_gt;
global camera_out;
global landmarks_gt;
global landmarks_out;
global landmarks_used;
global calib;
camera_gt = dlmread(strcat(blenddir, 'camera_poses.txt')); % each line is: frame_id,x,y,z,qx,qy,qz,qw
camera_out = zeros(size(camera_gt)); % each line is: frame_id,x,y,z,qx,qy,qz,qw
features_gt = dlmread(strcat(blenddir, 'tracks_dist.txt')); % each line is list of: landmark_id,feature_x,feature_y,feature_d,...
landmarks_gt = dlmread(strcat(blenddir, 'landmarks_3d.txt')); % each line is: x,y,z
landmarks_out = zeros(size(landmarks_gt)); % each line is: x,y,z,last_seen_frame_id
ateMat = zeros(size(camera_gt,1)-NUM_INITIALISE,1);
rpeMat = zeros(size(camera_gt,1)-NUM_INITIALISE,1);
addGPSPrior=false;
calcRPE_ATE = false;

poseVariance = zeros(size(camera_gt,1)-NUM_INITIALISE,1);


% This table stores the last usage of each landmark (frame_id, whenever the
% landmark was visible somewhere in the window).
% The structure is used for two purposes:
% 1st: Use landmarks only once, when building a window
% 2nd: Decide whether a loop occured, based on stored frame-IDs
landmarks_used = zeros(size(landmarks_gt,1),1);

if NUM_FRAMES < 1
    NUM_FRAMES = size(camera_gt, 1);
end
calib = Cal3_S2( ...
    634.8, ... % focal
    634.8, ... % focal
    0, ... % skew
    480,... % center
    270); % center

% Setup noise
measurementNoiseSigma = 2;
pointNoiseSigma = 3;
pointPriorNoise  = noiseModel.Isotropic.Sigma(3,pointNoiseSigma);
measurementNoise = noiseModel.Isotropic.Sigma(2,measurementNoiseSigma);

poseNoiseSigma =[.2 .2 .2]';
posePriorNoise = noiseModel.Diagonal.Sigmas(poseNoiseSigma);

% GPS Pose prior for frame 1, 250 and 500
rotG =Rot3.RzRyRx(0.0,0.0,0.0); % Don?t do anything with rotational pose
transG =Point3([2 2 2]'); % vraiance of 2m along x, y and z
gps_pose=Pose3(rotG,transG); % use this as a prior on camera pose for frame 1, 250 and 500

%% Sanity checks
assert(WINDOW_SIZE <= NUM_INITIALISE);
assert(NUM_FRAMES <= size(camera_gt,1));

%% Setup figure
figure;
% Optionally choose size according to your needs
%set(gcf,'units','pixel');
%set(gcf,'papersize',[960,960]);
%set(gcf,'paperpositionmode','auto');
subplot(2,1,1); % Top for map, bottom for video
axis equal;
hold on;
% Plot ground-truth trajectory as green line
plot3(camera_gt(:,2), camera_gt(:,3), camera_gt(:,4), 'g');

%% Initialise with ground-truth data
% Of course, this can not be done in real-life(!)
for i=1:NUM_INITIALISE
    fprintf('Loading ground-truth data for frame %d...\n', i);
    camera_out(i,:) = camera_gt(i,:);
    f = 1; % column of current feature ID
    while f < size(features_gt, 2) && features_gt(i,f) > 0
        feature_id = features_gt(i,f);
        % Initialise the point near ground-truth
        if ~in_map(feature_id)
            landmarks_out(feature_id,1:3) = landmarks_gt(feature_id,1:3);
            landmarks_used(feature_id,1) = NUM_INITIALISE;
        end
        f = f + 4;
    end
end

%% Add noise to 2D observations
if ADD_NOISE
    disp('Adding noise...')
    for i=1:NUM_FRAMES
        % 2D features
        f = 1;
        while f < size(features_gt, 2) && features_gt(i,f) > 0
            features_gt(i,f+1:f+2) = features_gt(i,f+1:f+2) + measurementNoiseSigma * randn(1,2);
            f = f + 4;
        end
    end
end


%% Visual-Odometry part: Optimise a window for each new frame
for i=NUM_INITIALISE+1:NUM_FRAMES
    
    tic % measure time
    
    %% Load image (for plotting only)
    subplot(2,1,2);
    hold off;
    img = imread(strcat(blenddir, '/frames/',sprintf('%04d',i),'.jpg'));
    imshow(img);
    subplot(2,1,1);
    
    %% Adding new frame
    window_start = i - WINDOW_SIZE;
    fprintf('Building graph for frame %d (Starting at %d)...\n', i, window_start);
    graph = NonlinearFactorGraph;
    initialEstimate = Values;
    
    landmarks_in_window = 0;
    redetected_landmarks = 0;
    for j=window_start:i
        
        if j==i
            % Initialise assuming constant motion model
            lPose = get_pose(camera_out,j-1);
            llPose = get_pose(camera_out,j-2);
            rPose = llPose.between(lPose);
            cam_pose = lPose.compose(rPose);
            
            % Initialise from last pose (method above is better)
            % cam_pose = get_pose(camera_out,j-1);
        else
            cam_pose = get_pose(camera_out,j);
        end
        
        % GTSAM does not ensure that the rotation component of a pose is
        % orthogonal, hence do this manually (important!)
        rdet = det(cam_pose.rotation.matrix);
        if abs(rdet-1) > 0.0001
            fprintf('Correcting R-det: %f \n', rdet);
            [U,S,V]=svd(cam_pose.rotation.matrix);
            % TODO, correct pose (single line missing)
            ortho = U*V';
            cam_pose = Pose3(Rot3(ortho),cam_pose.translation);
        end
        
        % TODO Initialise camera poses -- do not use positional priors though!
        
        initialEstimate.insert(symbol('p',j), cam_pose);
        
        if(addGPSPrior==true)
            %Adding prior for GPS Simulation
            if(i == 1 || i ==250 || i ==500)
                graph.add(PoseTranslationPrior3D(symbol('p',j), gps_pose, posePriorNoise));
                if(j==i)
                    redetected_landmarks=LOOP_THRESHOLD+1;
                    disp('Loop will close now');
                end
            end
        end
        
        % The following loop iterates over the (2D) features within frame 'j'
        f = 1; % column of current feature ID
        while f < size(features_gt, 2) && features_gt(j,f) > 0
            %            while f < 100 && features_gt(j,f) > 0
            
            feature_id = features_gt(j,f);
            
            % Initialise the point near ground-truth
            if landmarks_used(feature_id) ~= i
                
                if in_map(feature_id)
                    % This landmark is known to the system
                    % Was it seen in the current window or do we have a
                    % potential loop-closure?
                    if landmarks_used(feature_id) < window_start
                        redetected_landmarks = redetected_landmarks+1;
                        % Skip redetected landmarks and potentially close
                        % loop later
                        f = f + 4;
                        continue;
                    else
                        % The landmark was already seen, when building the window of frames
                        feature_pos = get_landmark(landmarks_out, feature_id);
                    end
                else
                    % "Triangulate"
                    rep = [(features_gt(j,f+1) - calib.px) / calib.fx, (features_gt(j,f+2) - calib.py) / calib.fy, 1] * features_gt(j,f+3);
                    rep = cam_pose.matrix * [rep'; 1];
                    feature_pos = Point3(rep(1),rep(2),rep(3));
                    
                    % Ground-Truth position: Don't do this, unless you are debugging!
                    % feature_pos = Point3(landmarks_gt(feature_id,1),landmarks_gt(feature_id,2),landmarks_gt(feature_id,3));
                end
                
                % Prior & initial estimate
                % TODO
                
                graph.add(PriorFactorPoint3(symbol('f',feature_id), feature_pos, pointPriorNoise));
                
                initialEstimate.insert(symbol('f',feature_id), feature_pos);
                
                landmarks_used(feature_id) = i;
                landmarks_in_window = landmarks_in_window + 1;
            end
            
            % TODO Measurements
            pt2 = Point2(features_gt(j,f+1),features_gt(j,f+2));
            graph.add(GenericProjectionFactorCal3_S2(pt2, measurementNoise, symbol('p',j), symbol('f',feature_id), calib));
            
            f = f + 4;
        end
        subplot(2,1,1);
    end
    
    %% Loop-closing
    if redetected_landmarks > LOOP_THRESHOLD
        % Build a more complex graph if a loop was detected
        disp('Loop detected! Building large graph...');
        
        % Add ALL landmarks seen so far
        disp('Adding all landmarks...');
        for l = 1:size(landmarks_used,1)
            % TODO
            if((landmarks_used(l) ~= i)&&(landmarks_used(l) ~= 0))
                lm = get_landmark(landmarks_out, l);
                graph.add(PriorFactorPoint3(symbol('f',l), lm, pointPriorNoise));
                initialEstimate.insert(symbol('f',l),lm );
                landmarks_used(l) = i;
            end
        end
        
        % Add measurements
        disp('Adding all poses & observations...');
        for j = 1:window_start-1
            % TODO
            cam_pose = get_pose(camera_out,j);
            if(addGPSPrior==true)
                if(j == 1 || j ==250 || j ==500)
                    graph.add(PoseTranslationPrior3D(symbol('p',j), gps_pose, posePriorNoise));
                end
            end
            initialEstimate.insert(symbol('p',j), cam_pose);
            f = 1; % column of current feature ID
            while f < size(features_gt, 2) && features_gt(j,f) > 0
                % TODO
                feature_id = features_gt(j,f);
                features = Point2(features_gt(j,f+1),features_gt(j,f+2));
                graph.add(GenericProjectionFactorCal3_S2(features, measurementNoise, symbol('p',j), symbol('f',feature_id), calib));
                f = f + 4;
            end
        end
        % The following code has to be informed that the window is bigger now
        window_start = 1;
    end
    
    optimizer = LevenbergMarquardtOptimizer(graph, initialEstimate);
    initialError = graph.error(initialEstimate);
    
    
    %% Plot ground-truth and initial estimate of markers (2D)
    subplot(2,1,2);
    hold on;
    plot_features(i, 'g+'); % ground-truth
    %plot_features(i, 'cx', cam_pose, landmarks_gt); % initial estimate
    subplot(2,1,1);
    %% Plot ground-truth and initial estimate in 3D
    % plot3(camera_gt(i,2),camera_gt(i,3),camera_gt(i,4), 'g+');
    % plot3(cam_pose.translation().x,cam_pose.translation().y,cam_pose.translation().z,'cx');
    %optimizer.optimizeSafely;
    last_error = initialError;
    for m=1:MAX_ITERATIONS
        optimizer.iterate();
        
        %% Print iteration information
        result = optimizer.values();
        error = graph.error(result);
        fprintf('Initial error: %f, %d.-iteration error: %f (%3.3f %%)\n', initialError, m, error, 100 * error/ initialError);
        
        %% Optionally, plot the movement and orientation of the pose during optimisation
        % (slow)
        % pose = result.atPose3(symbol('p', i));
        % pos = pose.translation();
        % mat = pose.matrix;
        % h = mat * [0;0;1;1]; % This point indicates the orientation of the orientation of the pose
        % h = [h(1:3)'; pos.x,pos.y,pos.z];
        % plot3(pos.x,pos.y,pos.z, 'bo');
        % plot3(h(:,1),h(:,2),h(:,3), 'b.');
        
        %% Optionally, plot the movement of features during optimisation
        % (slow)
        % subplot(2,1,2);
        % hold on;
        % plot_features(i, 'r.', pose, landmarks_gt);
        % subplot(2,1,1);
        
        % Break conditions
        if error < 0.001 %TODO find a nice value
            break
        end
        if last_error - error < 0.01%TODO find a nice value
            break
        end
        last_error = error;
    end
    
    % result = optimizer.values();
    % Optionally, retrieve marginals for plotting
    % marginals = Marginals(graph, result);
        
    error = graph.error(result);
    fprintf('Initial error: %f, Final error: %f (%3.3f %%)\n', initialError, error, 100 * error/ initialError);
    fprintf('Landmarks in window: %d\n', landmarks_in_window);
    
    % Read all poses that were optimised, 'pose' represents the most recent
    % frame afterwards
    for j=window_start:i
        pose = result.atPose3(symbol('p', j));
        pos = pose.translation();
        quat = pose.rotation().quaternion();
        camera_out(j,:) = [camera_gt(j,1) pos.x pos.y pos.z quat(2) quat(3) quat(4) quat(1)];
    end
    
    % Code to plot the variance of translational
    marginals = Marginals(graph, result);

    cov=marginals.marginalCovariance(symbol('p', i));
    diagCov = diag(cov);
    varX = diagCov(4);
    varY = diagCov(5);
    varZ = diagCov(6);
    poseVariance(i,1) = (varX^2+varY^2+varZ^2)^0.5;
    
    % Code to calculate and plot RPE and ATE calculated over each window
    if(calcRPE_ATE==true)
        % Overwrite in same file in each iteration the gt and estimated
        % poses of current window and then use these files for plotting RPE
        % and ATE
        
        dlmwrite('pose_curWindow.txt',camera_out(window_start:i,:),'delimiter','\t','precision',6);
        dlmwrite('pose_curWindow_gt.txt',camera_gt(window_start:i,:),'delimiter','\t','precision',6);
        
        [status ATE] = system('python ../external/tum-evaluation/evaluate_ate.py pose_curWindow_gt.txt pose_curWindow.txt');
        [status RPE] = system('python ../external/tum-evaluation/evaluate_rpe.py pose_curWindow_gt.txt pose_curWindow.txt --fixed_delta');
        
        ate(i-NUM_INITIALISE,1) = str2double(ATE);
        rpe(i-NUM_INITIALISE,1) = str2double(RPE);
    end
    
    % Read landmarks
    for l=1:length(landmarks_used)
        if landmarks_used(l) == i
            mark = result.atPoint3(symbol('f', l));
            landmarks_out(l,:) = [mark.x mark.y mark.z];
        end
    end
    
    %% Plotting
    
    % Plot markers on last frame
    if PLOT_FEATURES
        subplot(2,1,2);
        hold on;
        plot_features(i, 'rx', pose, landmarks_out);
        subplot(2,1,1);
    end
    
    % Plot trajectory
    try delete(tplt); end
    tplt = plot3(camera_out(1:i,2),camera_out(1:i,3),camera_out(1:i,4), 'r*');
    
    % Plot all landmarks
    if PLOT_LANDMARKS
        try delete(mplt); end
        mplt = plot3(landmarks_out(:,1),landmarks_out(:,2),landmarks_out(:,3), ...
            'Marker','.', 'Markersize',1, 'Color','black','LineStyle','none');
    end
    
    % Update figure and optionally save
    drawnow
    if SAVE_FIGURES
        saveas(gcf, strcat(outdir, sprintf('plot-%04d',i),'.png'));
    end
    
    toc
end

% Code to plot positional variance of poses
figure
plot(poseVariance(1:382,1));
ylabel('Magnitude of Variance of Translation of Camera Pose');
xlabel('Frames');

if(calcRPE_ATE==true)
    figure;
    plot(RPE)
    figure
    plot(ATE)
end
    
% Export results to hard-drive
disp('Exporting final results...');
dlmwrite(strcat(outdir, 'vo_output_poses.txt'),camera_out,'delimiter','\t','precision',6);

%% Some helper functions:

function p = get_pose(matrix, index)
import gtsam.*;
p = Pose3(Rot3.Quaternion(matrix(index,8), matrix(index,5), matrix(index,6), matrix(index,7)), ...
    Point3(matrix(index,2), matrix(index,3), matrix(index,4)));
end

function p = get_landmark(matrix, index)
import gtsam.*;
p = Point3(matrix(index,1),matrix(index,2),matrix(index,3));
end

% Returns true if a landmark was previously mapped, otherwise false
function t = in_map(landmark_index)
import gtsam.*;
global landmarks_used;
if landmarks_used(landmark_index) > 0
    t = true;
else
    t = false;
end
end

% Project 'landmarks' into the fhe frame with ID 'frame_id', using the 'camera_pose'
% If no pose or landmarks are provided, ground-truth feature-locations are plotted.
function plot_features(frame_id, style, camera_pose, landmarks)
import gtsam.*;
global calib;
global features_gt;

% Plot ground-truth if no pose or landmarks are provided
plot_gt = false;
if nargin < 3
    plot_gt = true;
else
    camera = SimpleCamera(camera_pose, calib);
end

% Iterate features
f = 1; % column of current feature ID
show_warning = false;
while f < size(features_gt, 2) && features_gt(frame_id,f) > 0
    if plot_gt
        plot(features_gt(frame_id,f+1),features_gt(frame_id,f+2),'g+');
    else
        feature_id = features_gt(frame_id,f);
        try
            point2d = camera.project(Point3(landmarks(feature_id,1),landmarks(feature_id,2),landmarks(feature_id,3)));
            plot(point2d.x,point2d.y,style);
        catch
            show_warning = true;
        end
    end
    f = f+4;
end
if show_warning
    warning('Could not plot all features.');
end
end
