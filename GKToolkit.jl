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
    return sum(a[n+1:end].*b[1:n])%0x02
end
#-------------------------------------------------
#-------------------------------------------------

function measurestabilizer!(stab,g,stabsign=nothing)
    #ignores sign
    n=size(stab,2)÷2

    noncummuting=findall(isodd,(@view stab[:,1:n])*g[n+1:2n].+@view(stab[:,n+1:2n])*g[1:n])

    if length(noncummuting)==0 return end

    pivotindex=noncummuting[1]


    if !isnothing(stabsign)
        for i in noncummuting[2:end]
            stabsign[i]⊻=productsign(stab[i,:],stab[pivotindex,:])⊻stabsign[pivotindex]
        end
        stabsign[pivotindex]=rand(0:1)
    end

    for i in noncummuting[2:end]
        stab[i,:].⊻=stab[pivotindex,:]
    end

    stab[pivotindex,:]=g

    return
end
#-------------------------------------------------
#-------------------------------------------------
function randomunitary(n)

    #generates a random n-qubit Clifford unitary.
    if n==1
        xz=randomxzpair(1)
        return reshape(xz,2,2)
    else

        xz=randomxzpair(n)
        uprime=extendtounitary(xz[1:2n],xz[2n+1:end])

        v=zeros(UInt8,2n,2n)
        v[1,1]=1
        v[n+1,n+1]=1
        v[[2:n;n+2:2n],[2:n;n+2:2n]]=randomunitary(n-1)


        return (uprime*v).%0x02
    end

end

#-------------------------------------------------
#-------------------------------------------------
function allcliffordunitaries(n)

    #generates all n-qubit Clifford unitaries. It's only viable for n<=3
    #though. after that it gets too large a set.

    if n==1
        us=[]
        p=possibleAssignments(4)
        for i in 1:size(p,1)
            if !commute(p[i,1:2],p[i,3:4])
                push!(us,reshape(p[i,:],2,2))
            end
        end
        return us
    else
        us=[]
        ws=allcliffordunitaries(n-1)
        v=zeros(UInt8,2n,2n)
        v[1,1]=1
        v[n+1,n+1]=1
        p=possibleAssignments(4n)
        for i in 1:size(p,1)
            if !commute(p[i,1:2n],p[i,2n+1:end])
                uprime=extendtounitary(p[i,1:2n],p[i,2n+1:end])
                for w in ws
                    v[[2:n;n+2:2n],[2:n;n+2:2n]]=w
                    u=(uprime*v).%0x02
                    push!(us,u)
                end
            end
        end
        return us
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------
function extendtounitary(xn,zn)
    #return a unitary which maps x1to the given xn & z1 to the given zn

    n=length(xn)÷2

    if commute(xn,zn)
        print("first vecotrs aren't anti-commuting!")
        return nothing
    end

    #we start by a complete basis for Paulis and replace a pair with xn,zn.
    #to choose which pair to replace we find a pair that appear non-trivially
    #in xn & zn.
    #this makes sure that the xbasis \cup zbasis form a complete set for Pauli strings on n qubits.

    xbasis=hcat(Matrix{UInt8}(1I,n,n),zeros(UInt8,n,n))
    zbasis=hcat(zeros(UInt8,n,n),Matrix{UInt8}(1I,n,n))

    dependentindex=findfirst(i->!commute(xn[[i,i+n]],zn[[i,i+n]]),1:n)
    xbasis[dependentindex,:]=xn
    zbasis[dependentindex,:]=zn
    z2swaprow!(xbasis,1,dependentindex)
    z2swaprow!(zbasis,1,dependentindex)


    #we now update other basises accordingly, such that [xi,xj]=[zi,zj]=0
    #and {xi,zj}=0

    for i=2:n

        #finding Zi, assuming the commutation relations are satisfied by
        #xj,zj pairs for j<i.


        #make sure Zi commutes with Xj for j<i
        for j=1:i-1
            if !commute(zbasis[i,:],xbasis[j,:])
                zbasis[i,:].⊻=zbasis[j,:]
            end
        end

        #make sure Zi commutes with Zj for j<i
        for j=1:i-1
            if !commute(zbasis[i,:],zbasis[j,:])
                zbasis[i,:].⊻=xbasis[j,:]
            end
        end


        #find an Xj in the remaining baisis vectors that doesn't commute with Zi
        #and name it Xi
        #first search among the remaing Xs

        foundaPair=false
        for j=i:n
            if !commute(zbasis[i,:],xbasis[j,:])
                z2swaprow!(xbasis,j,i)
                foundaPair=true
                break
            end
        end
        #if not found, search among the remaing Zs
        if foundaPair==false
            for j=i+1:n
                if !commute(zbasis[i,:],zbasis[j,:])
                    #swap the two
                    zbasis[j,:].⊻=view(xbasis,i,:)
                    xbasis[i,:].⊻=view(zbasis,j,:)
                    zbasis[j,:].⊻=view(xbasis,i,:)
                    foundaPair=true
                    break
                end
            end
        end

        #make sure Xi commutes with Xj for j<i
        for j=1:i-1
            if !commute(xbasis[i,:],xbasis[j,:])
                xbasis[i,:].⊻=zbasis[j,:]
            end
        end

        #make sure Xi commutes with Zj for j<i
        for j=1:i-1
            if !commute(xbasis[i,:],zbasis[j,:])
                xbasis[i,:].⊻=xbasis[j,:]
            end
        end

    end

    return transpose(vcat(xbasis,zbasis))
end

#-------------------------------------------------
#-------------------------------------------------
function applyunitary!(stab,u,indices)
    n=size(stab,2)÷2
    stab[:,[indices;indices.+n]]=(view(stab,:,[indices;indices.+n])*u').%0x02
    return
end
#-------------------------------------------------
#-------------------------------------------------

function applyunitary!(stab,u,indices,usign,stabsign)
    n=size(stab,2)÷2

    for i=1:size(stab,1)
        js=findall(isequal(0x01),@view stab[i,[indices;indices.+n]])
        if length(js)==0 continue end

        img=u[:,js[1]]
        sign=usign[js[1]]

        for j in js[2:end]
            sign⊻=usign[j]⊻ productsign(img,u[:,j])
            img.⊻=u[:,j]
        end

        stabsign[i]⊻=sign
        stab[i,[indices;indices.+n]]=img
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------
function randomxzpair(n)
    xz=rand([0x00,0x01],4n)
    while commute(xz[1:2n],xz[2n+1:4n])
        xz=rand([0x00,0x01],4n)
    end
    return xz
end
#-------------------------------------------------
#-------------------------------------------------

function commute(O1,O2)
    n=length(O1)÷2
    return (dot(O1[1:n],O2[n+1:2n])+dot(O1[n+1:2n],O2[1:n]))%2==0
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
function zproductstate(n)
    return hcat(zeros(UInt8,n,n),Matrix{UInt8}(I,n,n))
end
#-------------------------------------------------
#-------------------------------------------------
function entanglement(stab,indices)
    n=size(stab,1)
    return rank2!(stab[:,[indices;indices.+n]])-length(indices)
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
    clusterStabs=localStabs(stab,allparts)
    lstabs=zeros(UInt8,0,n)

    for part in partition
        lstabs=vcat(lstabs,localStabs(clusterStabs,setdiff(allparts,part)))
    end

    return rank2!(clusterStabs)-rank2!(lstabs)
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
