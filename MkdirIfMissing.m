function MkdirIfMissing(dir_pathname)

if ~exist(dir_pathname, 'dir')
    mkdir(dir_pathname);
end

end % end of function
