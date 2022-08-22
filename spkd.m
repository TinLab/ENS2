function d=spkd(tli,tlj,cost) 
% % d=spkd(tli,tlj,cost) calculates the "spike time" distance % (Victor & Purpura 1996) for a single cost % 
% tli: vector of spike times for first spike train 
% tlj: vector of spike times for second spike train 
% cost: cost per unit time to move a spike % 
% Copyright (c) 1999 by Daniel Reich and Jonathan Victor. 
% Translated to Matlab by Daniel Reich from FORTRAN code by Jonathan Victor. % 
nspi=length(tli); 
nspj=length(tlj); 
if cost==0 
    d=abs(nspi-nspj); 
    return 
elseif cost==Inf 
    d=nspi+nspj; 
    return 
end

% % INITIALIZE MARGINS WITH COST OF ADDING A SPIKE % 
% scr=zeros(nspi+1,nspj+1);
% scr(:,1)=(0:nspi)'; 
% scr(1,:)=(0:nspj); 
% if nspi & nspj 
%     for i=2:nspi+1 
%         for j=2:nspj+1 
%             scr(i,j)=min([scr(i-1,j)+1 scr(i,j-1)+1 scr(i-1,j-1)+cost*abs(tli(i-1)-tlj(j-1))]); 
%         end
%     end
% end
% d=scr(nspi+1,nspj+1);

% % Implementation of O(N) space complexity
scri = (0:nspi)';
scrj1 = (0:nspj);
scrj2 = (0:nspj);
if nspi & nspj
    for i=2:nspi+1
        scrj2 = zeros(size(scrj1));
        scrj2(1)=scri(i);
        for j=2:nspj+1
            scrj2(j)=min([scrj2(j-1)+1 scrj1(j)+1 scrj1(j-1)+cost*abs(tli(i-1)-tlj(j-1))]);
        end
        scrj1 = scrj2;
    end
end
d=scrj2(end);
