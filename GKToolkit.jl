using LinearAlgebra;
using Random;
using Distributed
using JLD2, FileIO, StatsBase, Dates
include("z2algebra.jl")


#-------------------------------------------------
#-------------------------------------------------
function productsign(a,b)
    #returns sign of a*b, when written in the standard form.
    n=length(a)÷2
    return sum(a[n+1:end].⊻b[1:n])%2
end
#-------------------------------------------------
#-------------------------------------------------

function measurestabilizer!(stab,g,stabsign=nothing)
    #ignores sign
    n=size(stab,1)

    gX=g[1:n]
    gZ=g[n+1:2n]

    stabX=view(stab,:,1:n)
    stabZ=view(stab,:,n+1:2n)

    noncummuting=findall(isodd,stabX*gZ+stabZ*gX)

    if length(noncummuting)==0 return end

    pivotindex=noncummuting[1]


    if !isnothing(stabsign)
        for i in noncummuting[2:end]
            stabsign[i]⊻=productsign(stab[i],stab[pivotindex])⊻stabsign[pivotindex]
        end
        stabsign[pivotindex]=rand(0:1)
    end

    for i in noncummuting[2:end]
        stabX[i,:].⊻=stabX[pivotindex,:]
        stabZ[i,:].⊻=stabZ[pivotindex,:]
    end

    stabX[pivotindex,:]=gX
    stabZ[pivotindex,:]=gZ

    return
end
#-------------------------------------------------
#-------------------------------------------------
function entanglement(stab,indices)
    n=size(stab,1)
    return rank2!(stab[:,[indices;indices.+n]])-length(indices)
end
#-------------------------------------------------
#-------------------------------------------------
function localStabs(stab,A)
    m,n=size(stab)
    N=n÷2

    A_c=setdiff(collect(1:N),A)
    r,M=rowecholen2_cm!(stab[:,[A_c;A_c.+N]])
    if r==m
        return zeros(Int8,0,n)
    else
        return (M*stab)[r+1:m,:].%2
    end
end

#-------------------------------------------------
#-------------------------------------------------
function independentStabs(stab)
    m,n=size(stab)
    N=n÷2

    r,M=rowecholen2_cm!(stab)
    return (M*stab)[1:r,:].%2
end

#-------------------------------------------------
#-------------------------------------------------
function negativity(stabs,A,B)
    lstabs=localStabs(stabs,union(A,B))
    J=JMatrix(lstabs,A)
    return rank2!(J)/2
end

#-------------------------------------------------
#-------------------------------------------------
function ghzyield(stab,partition)
    m,n=size(stab)


    allparts=union(partition...)
    # println(allparts)
    clusterStabs=localStabs(stab,allparts)
    lstabs=zeros(Int8,0,n)

##    println("clr:$(rank2!(copy(clusterStabs)))")
    for part in partition
        lstabs=vcat(lstabs,localStabs(clusterStabs,setdiff(allparts,part)))
 #       println("lr:$(rank2!(copy(lstabs)))")
    end
    r1=rank2!(clusterStabs)
    r2=rank2!(lstabs)
 #   println("r1:$r1, \t r2:$r2")
    return r1-r2
end
#-------------------------------------------------
#-------------------------------------------------
function ghzyield2(stab,partition)
    m,n=size(stab)

    allparts=union(partition...)
    clusterStabs=localStabs(stab,allparts)
    lstabs=zeros(Int8,0,n)


    for part in partition
        lstabs=independentStabs(vcat(lstabs,localStabs(clusterStabs,setdiff(allparts,part))))
    end
    return rank2!(localStabs(stab,allparts))-rank2!(lstabs)
end
#-------------------------------------------------
#-------------------------------------------------
function apply4qubitUnitary!(stab,u,i,j,k,l)
    #u.T is expected

    N=size(stab,1)
    stab[:,[i,j,k,l,i+N,j+N,k+N,l+N]]=(view(stab,:,[i,j,k,l,i+N,j+N,k+N,l+N])*u).%2


    return
end

#-------------------------------------------------
#-------------------------------------------------
function rowecholen2_cm!(A)
    #julia save arrays in the colum major format, so to improve performance, the matrix is transposed first, then turned into column echolen form.
    m,n=size(A)

    B=transpose(A)
    B=vcat(B,Matrix{Int64}(I,m,m))

    lastPivotCol=0

    for i in 1:n
        for j in (lastPivotCol+1):m
            if B[i,j]!=0
                for jj in (j+1):m
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


    return lastPivotCol,transpose(B)[:,n+1:end]
end

#-------------------------------------------------
#-------------------------------------------------
function gaussianelimination_row(A)
    AT,rank,pivotsT,MT=gaussianelimination_col(transpose(A))

    return transpose(AT),rank,[(j,i) for (i,j) in pivotsT],transpose(MT)
end

#-------------------------------------------------
#-------------------------------------------------



function rowecholen2_withM!(A)

    m,n=size(A)
    lastPivotRow=0
    A=hcat(A,Matrix{Int8}(I,m,m))

    for j in 1:n
        for i in (lastPivotRow+1):m
            if A[i,j]!=0
                for ii in (i+1):m
                    if A[ii,j]!=0
                        A[ii,j:end].⊻=A[i,j:end]
                    end
                end
                z2SwapRow!(A,i,lastPivotRow+1,j)
                lastPivotRow+=1
                break
            end
        end
    end

    return lastPivotRow,A[:,n+1:end]
end


#-------------------------------------------------
#-------------------------------------------------

function JMatrix(stabs,A)
    m,n=size(stabs)
    N=n÷2
    j=stabs[:,A]*transpose(stabs[:,A.+N])
    return (j+transpose(j)).%2
end

#-------------------------------------------------
#-------------------------------------------------
function checkStab(X)
    n=size(X,1)
    for i=1:n-1
        for j=i+1:n
            if !commute(X[i,:],X[j,:])
                println("not a stabilizer!! $i $j")
                return false
            end
        end
    end
    return true
end
#-------------------------------------------------
#-------------------------------------------------
function extendToUnitary(X_n,Z_n)
    n=length(X_n)÷2
    p_idx=pivotIdx(X_n,Z_n)

    if commute(X_n,Z_n)
        print("first vecotrs aren't anti-commuting!")
    end


    X_basis=hcat(Matrix{Int8}(1I,n,n),zeros(Int8,n,n))
    X_basis[p_idx,:]=X_n
    z2SwapRow!(X_basis,1,p_idx)

    Z_basis=hcat(zeros(Int8,n,n),Matrix{Int8}(1I,n,n))
    Z_basis[p_idx,:]=Z_n
    z2SwapRow!(Z_basis,1,p_idx)

    for i in 2:n
        #finding Zi

        #make sure Zi commutes with Xj for j<i
        for j in 1:i-1
            if !commute(Z_basis[i,:],X_basis[j,:])
                Z_basis[i,:].⊻=Z_basis[j,:]
            end
        end

        #make sure Zi commutes with Zj for j<i
        for j in 1:i-1
            if !commute(Z_basis[i,:],Z_basis[j,:])
                Z_basis[i,:].⊻=X_basis[j,:]
            end
        end


        #find an Xj that doesn't commute with Zi, name it Xi
        #first search among the remaing Xs
        foundaPair=false
        for j in i:n
            if !commute(Z_basis[i,:],X_basis[j,:])
                z2SwapRow!(X_basis,j,i)
                foundaPair=true
                break
            end
        end
        #if not found, search among the remaing Zs
        if foundaPair==false
            for j in i+1:n
                if !commute(Z_basis[i,:],Z_basis[j,:])
                    #swap the two
                    Z_basis[j,:].⊻=X_basis[i,:]
                    X_basis[i,:].⊻=Z_basis[j,:]
                    Z_basis[j,:].⊻=X_basis[i,:]
                    foundaPair=true
                    break
                end
            end
            if foundaPair==false
                println("no non-commuting partner found!")
            end
        end

        #make sure Xi commutes with Xj for j<i
        for j in 1:i-1
            if !commute(X_basis[i,:],X_basis[j,:])
                X_basis[i,:].⊻=Z_basis[j,:]
            end
        end

        #make sure Xi commutes with Zj for j<i
        for j in 1:i-1
            if !commute(X_basis[i,:],Z_basis[j,:])
                X_basis[i,:].⊻=X_basis[j,:]
            end
        end

    end

    return transpose(vcat(X_basis,Z_basis))
end

#-------------------------------------------------
#-------------------------------------------------
function pivotIdx(Sx,Sz)
    n=length(Sx)÷2
    for i in 1:n
        if !commute(Sx[[i,i+n]],Sz[[i,i+n]])
            return i
        end
    end
    return 0
end

#-------------------------------------------------
#-------------------------------------------------
function checkUnitarity(U)
    m,=size(U)
    ca=zeros(Int,m,m)
    for i in 1:m
        for j in 1:m
            ca[i,j]=commute(U[i,:],U[j,:]) ? 0 : 1
        end
    end
    return ca
end
#-------------------------------------------------
#-------------------------------------------------
function firstNZIdx(A)
    for j in eachindex(A)
        if A[j]!=0
            return j
        end
    end
    return 0
end
#-------------------------------------------------
#-------------------------------------------------
function lastNZIdx(A)
    for j in length(A):-1:1
        if A[j]!=0
            return j
        end
    end
    return 0
end
#-------------------------------------------------
#-------------------------------------------------

function commute(O1,O2)
    n=length(O1)÷2
    return (dot(O1[1:n],O2[n+1:2*n])+dot(O1[n+1:2*n],O2[1:n]))%2==0
end

#-------------------------------------------------
#-------------------------------------------------


function possibleAssignments(k)
    ns=zeros(Int8,2^k,k)
    for i in 1:2^k
        for j in 1:k
            ns[i,j]=(i÷2^(j-1))%2
        end
    end
    return ns
end
#-------------------------------------------------
#-------------------------------------------------

function generateAllCliffordUnitaries(n)
    #it returns them transposed now
    if n==1
        Us=[]
        p=possibleAssignments(4)
        for i in 1:size(p)[1]
            if !commute(p[i,1:2],p[i,3:4])
                push!(Us,reshape(p[i,:],2,2))
            end
        end
        return Us
    else
        Us=[]
        Vs=generateAllCliffordUnitaries(n-1)
        V=zeros(Int8,2*n,2*n)
        V[1,1]=1
        V[n+1,n+1]=1
        p=possibleAssignments(4*n)
        for i in 1:size(p,1)
            if !commute(p[i,1:2*n],p[i,2*n+1:end])
                U_prime=extendToUnitary(p[i,1:2*n],p[i,2*n+1:end])
                for v in Vs
                    V[[2:n;n+2:2*n],[2:n;n+2:2*n]]=v
                    U=(U_prime*V).%2
                    push!(Us,U)
                end
            end
        end
        return Us
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------
function ghzstabilizer(n)
    stab=zeros(UInt8,n,2n)
    for i=1:n-1
        stab[i,i+n]=1
        stab[i,i+1+n]=1
    end
    stab[n,1:n]=ones(UInt8,n)
    stabsign=zeros(UInt8,n)
    return stab,stabsign
end

#-------------------------------------------------
#-------------------------------------------------
function safe_savesnapshot(filename,fieldname,snapshot)
    jldopen("$(filename).mirror", "w") do file
        file[fieldname]=snapshot
    end
    cp("$(filename).mirror",filename,force=true)
    return
end
#-------------------------------------------------
#-------------------------------------------------
function safe_loadsnapshot(filename,fieldname)
    try
        jldopen(filename,"r") do file
            return file[fieldname]
        end
    catch
        cp("$(filename).mirror",filename,force=true)
        jldopen(filename,"r") do file
            return file[fieldname]
        end
    end
end
