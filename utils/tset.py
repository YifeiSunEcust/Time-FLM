# Definition for a binary tree node.
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        d = {}
        result = []
        for s in strs:
            pstr = sorted(s)
            newstr = ''.join(pstr)
            if newstr not in d:
                d[newstr]=[s]
            else:
                d[newstr].append(s)
        for i in d:
            result.append(d[i])
        return result
print(Solution().groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))