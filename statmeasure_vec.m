function sm = statmeasure_vec(vec)
%this function extracts the statistical features of a vector.
%mean
N = length(vec');
if  N(1) == 1
    sm = [0 0 0 0];
    return;
else
%mean
meanvec = mean(vec');
%variance
secmnt = var(vec');
%third moment or skewness;
skew = skewness(vec');
%fourth moment or kurtosis;
kurt = kurtosis(vec');
maxim = max(vec');

sm=[meanvec',secmnt',skew',kurt',maxim'];
end
return;