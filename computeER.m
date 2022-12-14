function ER = computeER(tli,tlj,win)

cost = spk_distance('par');
cost.timedelay = win;
cost.maxdelay = 1;

[d,desc] = spk_distance(tli,tlj,cost);

stats = desc.count;

ER = f1score(stats.falsen/stats.true, stats.falsed/stats.detections);

function ER = f1score(miss,falsep,nsptrue,nspest)
% function ER = f1score(nmiss,nfalsep,nsptrue,nspest)
% function ER = f1score(miss,falsep)

if nargin==0, help f1score, return, end

if nargin==4
    miss = float(miss)./float(nsptrue);
    falsep = float(falsep)./float(nspest);
end

miss(isnan(miss)) = 0; % no real spikes: no miss!
falsep(isnan(falsep)) = 0; % no spike detected: no false positive!
ss = 1-miss;
pr = 1-falsep;
ER = 1 - 2*ss.*pr./(ss+pr);
ER(ss+pr==0) = 1; % there are real spikes and detected spikes, but none of them do match!! 

function varargout = spk_distance(varargin)
% function [d desc] = spk_distance(tli,tlj[,cost])
% function [misses falsep] = spk_distance(tli,tlj[,cost],'basic')
% function cost = spk_distance('par')
%---
% Compute a distance and a more lengthy description of the differences
% between two spike trains.
%
% Input:
% - tli     vector of spike times for first spike train
% - tlj     vector of spike times for second spike train
% 
% Input/Output:
% - cost    structure defining the costs on which the distance is based
% 
% Output:
% - d       scalar - distance between spike trains
% - desc    structure - description (e.g. false positives, misses,
%           delays...)

% (Victor & Purpura 1996) for a single cost
% Copyright (c) 1999 by Daniel Reich and Jonathan Victor.
% Translated to Matlab by Daniel Reich from FORTRAN code by Jonathan Victor.
% Modified by Thomas Deneux and Tamas Lakner 2013


if nargin==0, help spk_distance, return, end

if nargin==1 && strcmp(varargin{1},'par')
    varargout = {defaultcost};
else
    [tli tlj] = deal(varargin{1:2});
    cost = defaultcost; dobasic = false;
    for i=3:nargin
        a = varargin{i};
        if isstruct(a)
            cost = fn_structmerge(defaultcost,a);
        elseif strcmp(a,'basic')
            dobasic = true;
        else
            error argument
        end
    end
    [d desc] = spikedistance(tli,tlj,cost);
    if dobasic
        counts = [desc.count];
        ntrue = sum([counts.true]);
        ndet = sum([counts.detections]);
        miss = sum([counts.falsen]);
        fp = sum([counts.falsed]);
        ntrue = max(ntrue,1); ndet = max(ndet,1); % avoid 0 divided by 0
        varargout = {miss/ntrue fp/ndet};
    else
        varargout = {d desc};
    end
end


%---
function cost = defaultcost

cost = struct( ...
    'miss0',        1, ...      % cost of isolated miss
    'miss1',        .5, ...     % cost of non-isolated miss
    'flsp0',        1, ...      % cost of isolated false-positive
    'flsp1',        .5, ...     % cost of non-isolated false-positive
    'nneighbors',   2, ...      % # neighbors that define a non-isolated spike
    'timeburst',    .5, ...     % time constant defining non-isolated spikes
    'timematch',    .0, ...     % maximal time distance for a perfect match
    'timedelay',    .5, ...     % maximal time distance for matching spikes
    'maxdelay',     2, ...      % cost of a match at maximal time distance
    'sides',        [] ...      % set the start and end times of the acquisition for a 'smart' count that minimizes side effects
    );

%---
function [d desc] = spikedistance(tli,tlj,cost)

% multiple data
if iscell(tli)
    n = numel(tli);
    if ~iscell(tlj) || numel(tlj)~=n, error 'multiple data do not match', end
    for i=1:n
        costi = cost;
        if ~isempty(cost.sides) && ~isvector(cost.sides)
            costi.sides = cost.sides(i,:);
        end
        [d(i) desc(i)] = spikedistance(tli{i},tlj{i},costi); %#ok<AGROW> 
    end
    d = sum(d);
    return
end

tli = row(tli); tlj = row(tlj);
nspi=length(tli); % start spikes
nspj=length(tlj); % target spikes

% which estimated (target) spikes are isolated or not
timedistances = abs(fn_subtract(row(tlj),column(tli)));
neighbors = (timedistances<cost.timeburst);
nonisolatedj = sum(neighbors,1)>=cost.nneighbors;
falsepcost = nonisolatedj*cost.flsp1 + ~nonisolatedj*cost.flsp0;

% which real (start) spikes are isolated or not
timedistances = abs(fn_subtract(tli(:)',tli(:)));
neighbors = (timedistances<cost.timeburst);
nonisolatedi = sum(neighbors,1)>=cost.nneighbors;
misscost = nonisolatedi*cost.miss1 + ~nonisolatedi*cost.miss0;

% cost matrix
costbuild = Inf(nspi+1,nspj+1);

% where do we come from: 1 = miss; 2 = false positive; 3 = shift
map = zeros(nspi+1,nspj+1);

% initialize margins with cost of adding a spike
    % costbuild(1:length(misscost)+1,1) = [0; cumsum(misscost)'];
    % costbuild(1,1:length(falsepcost)+1) = [0 cumsum(falsepcost)];
costbuild(:,1) = [0; cumsum(misscost)'];
costbuild(1,:) = [0 cumsum(falsepcost)];


% doubly recursive calculation of cost

% if nspi && nspj
%    for i=2:nspi+1
%       for j=2:nspj+1
%          [costbuild(i,j) map(i,j)] = min([ ...
%              costbuild(i-1,j) + misscost(i-1) ...
%              costbuild(i,j-1) + falsepcost(j-1) ...
%              costbuild(i-1,j-1) + shiftcost(abs(tli(i-1)-tlj(j-1)),cost) ...
%              ]);
%       end
%    end
% end

timedistances = abs(fn_subtract(tli(:),tlj(:)'));
shiftcosts = shiftcost(timedistances,cost);
okdistance = (timedistances<cost.timedelay);
if nspi && nspj
    for i=2:nspi+1
        jokstart = find(okdistance(i-1,:),1,'first')+1;
        jokend = find(okdistance(i-1,:),1,'last')+1;
        if isempty(jokstart), jokstart = nspj+2; jokend = nspj+1; end
        costbuild(i,1:jokstart-1) = costbuild(i-1,1:jokstart-1) + misscost(i-1);
        map(i,1:jokstart-1) = 1;
        for j = jokstart:jokend
            [costbuild(i,j) map(i,j)] = min([ ...
                costbuild(i-1,j) + misscost(i-1) ...
                costbuild(i,j-1) + falsepcost(j-1) ...
                costbuild(i-1,j-1) + shiftcosts(i-1,j-1) ...
                ]);
        end
        costbuild(i,jokend+1:end) = costbuild(i,jokend) + cumsum(falsepcost(jokend:end));
        map(i,jokend+1:end) = 2;
    end
end

d=costbuild(nspi+1,nspj+1);

% Reconstruction of the match
desc = struct( ...
    'miss0',    [], ...
    'miss1',    [], ...
    'falsep0',  [], ...
    'falsep1',  [], ...
    'perfectshift', zeros(2,0), ...
    'okshift',      zeros(2,0) ...
    );
curpos = [nspi+1 nspj+1];
while true
    switch map(curpos(1),curpos(2))
        case 1
            curpos(1) = curpos(1)-1;
            if nonisolatedi(curpos(1))
                desc.miss1(end+1) = curpos(1);
            else
                desc.miss0(end+1) = curpos(1);
            end
        case 2
            curpos(2) = curpos(2)-1;
            if nonisolatedj(curpos(2))
                desc.falsep1(end+1) = curpos(2);
            else
                desc.falsep0(end+1) = curpos(2);
            end
        case 3
            curpos = curpos-1;
            if abs(tli(curpos(1))-tlj(curpos(2)))<cost.timematch
                desc.perfectshift(:,end+1) = curpos';
            else
                desc.okshift(:,end+1) = curpos';
            end
    end
    if any(curpos==1), break, end
end

% More statistics
% (indices for miss/false positive/match)
if curpos(1)>1
    desc.miss0 = [desc.miss0 find(~nonisolatedi(1:curpos(1)-1))];
    desc.miss1 = [desc.miss1 find(nonisolatedi(1:curpos(1)-1))];
elseif curpos(2)>1
    desc.falsep0 = [desc.falsep0 find(~nonisolatedj(1:curpos(2)-1))];
    desc.falsep1 = [desc.falsep1 find(nonisolatedj(1:curpos(2)-1))];
end
desc.miss = sort([desc.miss0 desc.miss1]);
desc.falsep = sort([desc.falsep0 desc.falsep1]);
desc.shift = [desc.perfectshift desc.okshift];
% (counts)
F = fieldnames(desc);
desc.count = struct;
for i=1:length(F)
    f = F{i};
    desc.count.(f) = size(desc.(f),2);
end
% (smart counts: minimize side effects)
if ~isempty(cost.sides)
    ts = cost.sides(1) + cost.timedelay;
    te = cost.sides(2) - cost.timedelay;
    truep  = tli(desc.shift(1,:));
    falsen = tli(desc.miss);
    trued  = tlj(desc.shift(2,:));
    falsed = tlj(desc.falsep);
    desc.count.truep  = sum(truep>=ts & truep<=te);
    desc.count.falsen = sum(falsen>=ts & falsen<=te);
    desc.count.trued  = sum(trued>=ts & trued<=te);
    desc.count.falsed = sum(falsed>=ts & falsed<=te);
else
    desc.count.truep  = desc.count.shift;
    desc.count.falsen = desc.count.miss;
    desc.count.trued  = desc.count.shift;
    desc.count.falsed = desc.count.falsep;
end
desc.count.true = desc.count.truep + desc.count.falsen;
desc.count.detections = desc.count.trued + desc.count.falsed;
% (delays)
if isempty(desc.shift)
    desc.delays = [];
else
    desc.delays = (tlj(desc.shift(2,:))-tli(desc.shift(1,:)));
end

%---
function c = shiftcost(deltat,cost)

% if deltat<cost.timematch
%     c = 0;
% elseif deltat<cost.timedelay
%     c = (deltat-cost.timematch)/(cost.timedelay-cost.timematch) * cost.maxdelay;
% else
%     error programming
% end

c = zeros(size(deltat));
infcost = (deltat>cost.timedelay);
c(infcost) = Inf;
okcost = (deltat>cost.timematch & ~infcost);
c(okcost) = (deltat(okcost)-cost.timematch)/(cost.timedelay-cost.timematch) * cost.maxdelay;

function s = fn_structmerge(s,varargin)
% function s = fn_structmerge(s,s1[,'skip|strict'][,'recursive'][,'type'][,'i'])
%---
% set or replace values in s from those in s1, where s and s1 are
% structures of the same size
% - if 'skip', or 'strict' flag is specified: does not add new field in
%   structure s (generates error if 'strict' flag and s1 has additional
%   fields)
% - if 'recursive' flag, field values which are themselves structures are
%   not merely replaced, but are also merged using fn_structmerge 
% - if 'type' flag is specified: also requires field values to be the same
%   class in s and s1 when the field already exists in s (generates error
%   if it is not the case, except that it performs the conversions
%   0/1->false/true  and char array->cell array of strings)
% - 'i' flag for 'case insensitive': merge together field names that might
%   differ in case
%
% See also fn_structcat

% Thomas Deneux
% Copyright 2007-2017

% Input
if nargin>=2 && isstruct(varargin{1})
    s1 = varargin{1};
    varargin(1)=[];
else
    s1 = struct(varargin{:});
end
[skip strict recursive type caseinsensitive] = deal(false);
i=0;
while i<length(varargin)
    i = i+1;
    a = varargin{i};
    switch a
        case 'skip'
            skip = true;
        case 'strict'
            strict = true;
        case 'recursive'
            recursive = true;
        case 'type'
            type = true;
        case 'i'
            caseinsensitive = true;
        otherwise
            i = i+1;
            s1.(a) = varargin{i};
    end
end
if isempty(s1), return, end
if any(size(s1)~=size(s))
    if isempty(s)
        s = repmat(fn_structinit(s),size(s1));
    elseif isscalar(s)
        s = repmat(s,size(s1));
    elseif isscalar(s1)
        s1 = repmat(s1,size(s));
    else
        error('size mismatch')
    end
end
skip = skip | strict;

F = fieldnames(s);
F1 = fieldnames(s1);
for k=1:length(F1)
    f1 = F1{k};
    if caseinsensitive
        idx = find(strcmpi(f1,F));
        if isscalar(idx), f=F{idx}; else f=f1; end
    else
        f = f1;
    end
    if skip && ~isfield(s,f)
        if strict
            error('field ''%s'' not present in original structure',f1)
        end
    elseif recursive && isfield(s,f) && isstruct(s(1).(f))
        for i=1:numel(s)
            if isscalar(s1), j=1; else j=i; end
            val = s1(j).(f1);
            if isstruct(val)
                s(i).(f) = fn_structmerge(s(i).(f),val,varargin{:});
            else
                if strict, error('value for field ''%s'' should be a structure',f), end
                s(i).(f) = val;
            end
        end
    else
        for i=1:numel(s)
            if isscalar(s1), j=1; else j=i; end
            if type && isfield(s,f) && ~strcmp(class(s(i).(f)),class(s1(j).(f1)))
                if islogical(s(i).(f)) && isnumeric(s1(j).(f1)) ...
                        && isscalar(s1(j).(f1)) && ismember(s1(j).(f1),[0 1])
                    s(i).(f) = logical(s1(j).(f1));
                elseif iscell(s(i).(f)) && (isempty(s(i).(f)) || ischar(s(i).(f){1})) ...
                        && ischar(s1(i).(f1))
                    s(i).(f) = cellstr(s1(i).(f1));
                else
                    error('class mismatch')
                end
            else
                s(i).(f) = s1(j).(f1);
            end
        end
    end
end

function y=fn_subtract(u,v)
% function y=fn_subtract(u,v)
%----
% tool to subtract a matrix row- or column-wise
% ex: y = fn_subtract(rand(3,4),(1:3)')
%     y = fn_subtract(rand(5,2,5),shiftdim(ones(5,1),-2))
%
% See also fn_add, fn_mult, fn_div

% Thomas Deneux
% Copyright 2012-2017

y = bsxfun(@minus,u,v);

function x = row(x)
% function x = row(x)
%---
% reshape x to a row vector
%
% See also column, matrix, third, fourth

% Thomas Deneux
% Copyright 2015-2017

x = x(:)';

function x = column(x)
% function x = column(x)
%---
% reshape x to a column vector
%
% See also row, matrix, third, fourth

% Thomas Deneux
% Copyright 2015-2017

x = x(:);