function rank2!(A)
    m,n=size(A)
    lastpivotcol=0

    for i in 1:m
        for j in (lastpivotcol+1):n
            if A[i,j]!=0
                for jj in (j+1):n
                    if A[i,jj]!=0
                        A[i:end,jj].⊻=view(A,i:end,j)
                    end
                end
                z2swapcol!(A,j,lastpivotcol+1,i)
                lastpivotcol+=1
                break
            end
        end
    end

    return lastpivotcol
end
#-------------------------------------------------
#-------------------------------------------------
function z2swaprow!(A,i,j,leftcol=1)
    if i!=j
        A[i,leftcol:end].⊻=view(A,j,leftcol:end)
        A[j,leftcol:end].⊻=view(A,i,leftcol:end)
        A[i,leftcol:end].⊻=view(A,j,leftcol:end)
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------
function z2swapcol!(A,i,j,toprow=1)
    if i!=j
        A[toprow:end,i].⊻=view(A,toprow:end,j)
        A[toprow:end,j].⊻=view(A,toprow:end,i)
        A[toprow:end,i].⊻=view(A,toprow:end,j)
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------
function gaussianelimination_col(A)
    m,n=size(A)
    B=vcat(A,Matrix{Uint8}(I,n,n))
    pivots=[]
    lastPivotCol=0

    for i in 1:m
        for j in (lastPivotCol+1):n
            if B[i,j]!=0
                push!(pivots,(i,lastPivotCol+1))
                for jj in (j+1):n
                    if B[i,jj]!=0
                        B[i:end,jj].⊻=B[i:end,j]
                    end
                end
                z2SwapCol!(B,j,lastPivotCol+1,i)
                lastPivotCol+=1
                break
            end
        end
    end

    rank=lastPivotCol
    return B[1:m,:],rank,pivots,B[m+1:end,:]
end
#-------------------------------------------------
#-------------------------------------------------
