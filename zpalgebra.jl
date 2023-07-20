#-------------------------------------------------
#-------------------------------------------------
function swapcol!(A,i,j,toprow=1)
    #swap columns i and j
    #if toprow is given it only swaps the A[toprow:end,i] with A[toprow:end,j]
    #doesn't depend on Zp algebra

    if i!=j
        temp=A[toprow:end,i]
        A[toprow:end,i]=@view A[toprow:end,j]
        A[toprow:end,j]=temp
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------
function swaprow!(A,i,j,leftcol=1)
    #swap rows i and j
    #if toprow is given it only swaps the A[i,leftcol:end] with A[j,leftcol:end]
    #doesn't depend on Zp algebra

    if i!=j
        temp=A[i,leftcol:end]
        A[i,leftcol:end]=@view A[j,leftcol:end]
        A[j,leftcol:end]=temp
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------

function extendedEuclideanAlg(a,b) 
    r0=a
    r1=b

    s0=1
    s1=0

    t0=0
    t1=1


    r=r0%r1

    q=(r0-r)÷r1
    s=s0-q*s1
    t=t0-q*t1

    while r != 0
        temp=r1
        r1=r
        r0=temp

        r=r0%r1
        q=(r0-r)÷r1

        s0=s1
        s1=s
        s=s0-q*s1

        t0=t1
        t1=t
        t=t0-q*t1
        
    end

    return r1,s1,t1
end
#-------------------------------------------------
#-------------------------------------------------


function extendedEuclideanAlg(a,b,p) 
    T=typeof(a)

    r0=a
    r1=b

    s0=T(1)
    s1=T(0)

    t0=T(0)
    t1=T(1)


    r=r0%r1



    q=(r0-r)÷r1
    
    s=mod(s0-q*s1,p)
    t=mod(t0-q*t1,p)



    while r != 0
        temp=r1
        r1=r
        r0=temp

        r=r0%r1
        q=(r0-r)÷r1

        s0=s1
        s1=s
        
        t0=t1
        t1=t
        
        s=mod(s0-q*s1,p)
        t=mod(t0-q*t1,p)
        
    end

    return r1,s1,t1
end
#-------------------------------------------------
#-------------------------------------------------

function inverse(a,p)
    return extendedEuclideanAlg(a,p,p)[2]
end

#-------------------------------------------------
#-------------------------------------------------




function rank_p!(A,p)
    #find rank(A) in Z_p
    m,n=size(A)
    lastpivotcol=0

    for i in 1:m
        for j in (lastpivotcol+1):n
            if A[i,j]!=0
                for jj in (j+1):n
                    if A[i,jj]!=0
                        A[i:end,jj]=@views mod.( A[i:end,jj].+  A[i:end,j].*(inverse(A[i,j],p)*(p-A[i,jj])),p)
                    end
                end
                swapcol!(A,j,lastpivotcol+1,i)
                lastpivotcol+=1
                break
            end
        end
    end

    return lastpivotcol
end

#-------------------------------------------------
#-------------------------------------------------
# function gaussianelimination_col(A)
#     m,n=size(A)
#     B=vcat(A,Matrix{Uint8}(I,n,n))
#     pivots=[]
#     lastPivotCol=0

#     for i in 1:m
#         for j in (lastPivotCol+1):n
#             if B[i,j]!=0
#                 push!(pivots,(i,lastPivotCol+1))
#                 for jj in (j+1):n
#                     if B[i,jj]!=0
#                         B[i:end,jj].⊻=B[i:end,j]
#                     end
#                 end
#                 z2SwapCol!(B,j,lastPivotCol+1,i)
#                 lastPivotCol+=1
#                 break
#             end
#         end
#     end

#     rank=lastPivotCol
#     return B[1:m,:],rank,pivots,B[m+1:end,:]
# end
# #-------------------------------------------------
# #-------------------------------------------------
# function gaussianelimination_row(A)
#     AT,rank,pivotsT,MT=gaussianelimination_col(transpose(A))

#     return transpose(AT),rank,[(j,i) for (i,j) in pivotsT],transpose(MT)
# end

#-------------------------------------------------
#-------------------------------------------------
function allPossibleBitAssignments(k)
    #returns  all possible bit strings of k bits, arranged in a k x 2^k matrix.
    # the j'th column for 1<= j <= 2^k is bassically j-1 written in binary
    ns=zeros(UInt8,k,2^k)
    for i in 1:k, j in 0:2^k-1
        
            ns[i,j+1]=(j÷2^(i-1))%2
    end
    return ns
end
