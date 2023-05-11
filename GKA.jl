using LinearAlgebra;
using Random;
using Distributed
using JLD2, FileIO, StatsBase, Dates
include("z2algebra.jl")


#-------------------------------------------------
#-------------------------------------------------
mutable struct StateDensityMatrix
    #save the stabs in column major format. each column is a Pauli string. first n rows represent the X part
    #first r=rank columns are the stabilizers. the destabilizer associated to the i'th stabilizer is in n+i'th column.
    tab::Array{UInt8,2}
    phases::Array{UInt8}
    rank::Int
end
#-------------------------------------------------
#-------------------------------------------------
function productSign(a,b)
    
    #returns sign of a*b, when written in the standard form.
    #*0x02 is needed since we represent -1 as 0x02 (0x01 is imaginary i)
    
    n=length(a)÷2
    return (sum(a[n+1:end].*b[1:n])%0x02)*0x02
    
end
#-------------------------------------------------
#-------------------------------------------------
function commute(g::Array{UInt8,1},p::Array{UInt8,1})
    #whether g and p commute    
    n=length(g)÷2
    return sum(g[1:n].*p[n+1:2n]+g[n+1:2n].*p[1:n])%0x02
    
end
#-------------------------------------------------
#-------------------------------------------------
function commute(tab::Array{UInt8,2},g::Array{UInt8,1})
    #whether columns of tab commute with g.
    n=length(g)÷2
    return (transpose(tab[n+1:2n,:])*g[1:n].+transpose(tab[1:n,:])*g[n+1:2n]).%0x02
    
end
#-------------------------------------------------
#-------------------------------------------------
function addCols(state::StateDensityMatrix,i,j)
    
    #replaces g_j with gi*g_j in the tableau. takes care of phases
    
    n=size(state.tab,1)÷2
    state.phases[j]=(state.phases[j]+state.phases[i]+productSign(state.tab[:,j],state.tab[:,i]))%0x04
    state.tab[:,j]=state.tab[:,j].⊻state.tab[:,i]
    return
    
end
#-------------------------------------------------
#-------------------------------------------------
function addCols(stab::Array{UInt8,2},i,j)
    
    #replaces g_j with gi*g_j in the stab matrix.
    
    stab[:,j]=stab[:,j].⊻ stab[:,i]
    return
    
end
#-------------------------------------------------
#-------------------------------------------------
function swapCols(state::StateDensityMatrix,i,j)
    
    #swaps stabilizer i and j. takes care of phases.
    
    temp_col=state.tab[:,i]
    temp_phase=state.phases[i]
    
    state.tab[:,i]=state.tab[:,j]
    state.phases[i]=state.phases[j]
    
    state.tab[:,j]=temp_col
    state.phases[j]=temp_phase
    
    return
end
#-------------------------------------------------
#-------------------------------------------------
function swapCols(stab::Array{UInt8,2},i,j)
    
    #swaps columns i and j in a matrix stab.
    
    temp_col=stab[:,i]
    
    stab[:,i]=stab[:,j]
    stab[:,j]=temp_col
    return stab
end
#-------------------------------------------------
#-------------------------------------------------
function updateDestabilizers(state::StateDensityMatrix,i)
#     this function assumes i<=n
#     it also assumes that state.tab has a valid form except for some js with j>n and j!=i+n
#     ignores phases becasue the phase of destabilizers is irrelavant.
    n=size(state.tab,1)÷2
    
    commuteSign=commute(state.tab[:,n+1:2n],state.tab[:,i])
    commuteSign[i]=0x00
            
    noncommuting=findall(x->x==0x01,commuteSign).+n
    for j in noncommuting
        addCols(state,i+n,j)
    end
    
    return
end
#-------------------------------------------------
#-------------------------------------------------
function insertStabilizer(state::StateDensityMatrix,g,gphase,i)
#     moves the i'th stabilizer to the i'th destabilizer position and puts g as the i'th stabilizzer
    n=size(state.tab,1)÷2
    state.tab[:,i+n]=state.tab[:,i]
    state.tab[:,i]=g
    state.phases[i]=gphase
end
#-------------------------------------------------
#-------------------------------------------------
function measure(state::StateDensityMatrix,g,gphase=0x00,verbose=false; evaluate_outcome=false)
    verbose && printstyled("\n measuring: $g\n";color=:blue)
    verbose && println("state has originally tableau:")
    verbose && display(state.tab)
    verbose && println("with rank:$(state.rank) and phases:$(state.phases)")
    
    n=size(state.tab,1)÷2
    r=state.rank
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],g))

    if length(noncommuting)==0
        verbose && println("no noncommuting in-group stabilizer")
        #check whether g is in the stabilizer group
        
        noncommuting_S=findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
        if length(noncommuting_S)!=0
            verbose && println("a non-commuting out-of-group stabilizer has been found:\n $(state.tab[:,noncommuting_S[1]])")

            #a non-trivial operator has been measured. 
            
            pivotIdx=noncommuting_S[1]
            
            #after this loop we have a valid table with a single stabilizer that anti-commutes with g
            for j in noncommuting_S[2:end]
                addCols(state,pivotIdx,j)
#                 the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#                 addCols(state,j+n,pivotIdx+n)     
            end

            outcome=rand([0x00,0x02])
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
                        
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group stabilizer")
        
        noncommuting_D=findall(x->x==0x01,commute(state.tab[:,r+1+n:2n],g)).+(r+n)
        if length(noncommuting_D)!=0
            verbose && println("a non-commuting out-of-group destabilizer has been found:\n $(state.tab[:,noncommuting_D[1]])")
#             a non-trivial operator has been measured. 
#             we swap the stabilizer and destabilizer for the first non-commuting destabilizer
#             By doing so, we arrive at a valid tableau and a single stabilizer that anti-commutes with g
            swapCols(state,noncommuting_D[1],noncommuting_D[1]-n)
            pivotIdx=noncommuting_D[1]-n
            
            outcome=rand([0x00,0x02])
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
            
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)            
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group destabilizer.\n g is in the stabilizer group")

        
#         +-g is in the stabilizer group, so the measurement outcome is deterministic and rank remains the same
        if evaluate_outcome==false
            return
        end
        noncommuting_DS=findall(x->x==0x01,commute(state.tab[:,1+n:r+n],g))
        if length(noncommuting_DS)==0
            if sum(g)==0
                return 1
            else
                printstyled("ERROR: SOMETHING IS VERY WRONG: g commutes with everything!\n";color = :red)
                println("g:")
                display(g)
                println("tab:")
                display(state.tab)
                println("rank:$(state.rank)")
                return
            end
        end
        ph=state.phases[noncommuting_DS[1]]
        op=state.tab[:,noncommuting_DS[1]]
        
        for i in noncommuting_DS[2:end]
            ph=ph+state.phases[i]+productSign(op,state.tab[:,i])
            op=op.⊻state.tab[:,i]
        end
        ph=ph%0x04
        
        if g!=op
            
            printstyled("ERROR: SOMETHING IS VERY WRONG:\n $g \n is not equal to\n $op\n";color = :red)
            println("g:")
            display(g)
            println("tab:")
            display(state.tab)
            println("rank:$(state.rank)")
        end
        if gphase==ph
            return 0x00
        elseif gphase==(ph+0x02)%0x04
            return 0x02
        else
            printstyled("ERROR: SOMETHING IS VERY WRONG. MAYBE YOU ARE MEASURING AN ANTI-HERMITIONA OPERATOR\n";color = :red)
            return 
        end
            
    else
        verbose && println("g anti-commutes with a stabilizer: $(state.tab[:,noncommuting[1]])")
        #the measurement is not compatible with the stabilizer group. the outcome is random
        pivotIdx=noncommuting[1]
        for j in noncommuting[2:end]
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end
        
        #update the out-of-group stabilizers, so tehy commute with the newly added stabilizer
        for j in findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end

        outcome=rand([0x00,0x02])
        insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
        updateDestabilizers(state,pivotIdx)
        return outcome
    end
end
#-------------------------------------------------
#-------------------------------------------------

function forceMeasure(state,g,gphase=0x00,verbose=false)
#     the only deference between this function and measure is:
#     1) the phase of g will not be randomized when g is not in the stabilizer group.
#     2) if -gphase*g is in the stabilizer group, it kills the state (sets it to zero) 


    
    verbose && printstyled("\n projecting on: $g\n with phase:$gphase ";color=:blue)
    verbose && println("state's tableau:")
    verbose && display(state.tab)
    verbose && println("rank:$(state.rank), phases:$(state.phases)")

    
    n=size(state.tab,1)÷2
    r=state.rank
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],g))

    if length(noncommuting)==0
        verbose && println("no noncommuting in-group stabilizer")
        #check whether g is in the stabilizer group
        
        noncommuting_S=findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
        if length(noncommuting_S)!=0
            verbose && println("a non-commuting out-of-group stabilizer has been found:\n $(state.tab[:,noncommuting_S[1]])")

            #a non-trivial operator has been measured. 
            
            pivotIdx=noncommuting_S[1]
            
            #after this loop we have a valid table with a single stabilizer that anti-commutes with g
            for j in noncommuting_S[2:end]
                addCols(state,pivotIdx,j)
#                 the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#                 addCols(state,j+n,pivotIdx+n)     
            end

#             since it is forced measurement
#             outcome=rand([0x00,0x02])
            outcome=0x00
            
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
                        
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group stabilizer")
        
        noncommuting_D=findall(x->x==0x01,commute(state.tab[:,r+1+n:2n],g)).+(r+n)
        if length(noncommuting_D)!=0
            verbose && println("a non-commuting out-of-group destabilizer has been found:\n $(state.tab[:,noncommuting_D[1]])")
#             a non-trivial operator has been measured. 
#             we swap the stabilizer and destabilizer for the first non-commuting destabilizer
#             By doing so, we arrive at a valid tableau and a single stabilizer that anti-commutes with g
            swapCols(state,noncommuting_D[1],noncommuting_D[1]-n)
            pivotIdx=noncommuting_D[1]-n
            
#             since it is forced measurement
#             outcome=rand([0x00,0x02])
            outcome=0x00
            
            insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
            updateDestabilizers(state,pivotIdx)
            
            swapCols(state,pivotIdx,r+1)
            swapCols(state,pivotIdx+n,r+1+n)            
            
            state.rank=r+1
            return outcome
        end
        verbose && println("no noncommuting out-of-group destabilizer.\n g is in the stabilizer group")

        
#         +-g is in the stabilizer group, so the measurement outcome is deterministic and rank remains the same
        noncommuting_DS=findall(x->x==0x01,commute(state.tab[:,1+n:r+n],g))
        if length(noncommuting_DS)==0
            if sum(g)==0
                return 1
            else
                printstyled("ERROR: SOMETHING IS VERY WRONG: g commutes with everything!\n";color = :red)
                println("g:")
                display(g)
                println("tab:")
                display(state.tab)
                println("rank:$(state.rank)")
                return
            end
        end
        ph=state.phases[noncommuting_DS[1]]
        op=state.tab[:,noncommuting_DS[1]]
        
        for i in noncommuting_DS[2:end]
            ph=ph+state.phases[i]+productSign(op,state.tab[:,i])
            op=op.⊻state.tab[:,i]
        end
        ph=ph%0x04
        
        if g!=op
            
            printstyled("ERROR: SOMETHING IS VERY WRONG:\n $g \n is not equal to\n $op\n";color = :red)
            println("g:")
            display(g)
            println("tab:")
            display(state.tab)
            println("rank:$(state.rank)")
        end
        if gphase==ph
            return 0x00
        elseif gphase==(ph+0x02)%0x04
                printstyled("WARNING: THE STATE HAD NO AMPLITITUDE IN THE FORCED SUBSPACE. NOTHING WILL BE DONE\n";color = :red)
            return 0x00
        else
            printstyled("ERROR: SOMETHING IS VERY WRONG. MAYBE YOU ARE MEASURING AN ANTI-HERMITIONA OPERATOR\n";color = :red)
            return 
        end
            
    else
        verbose && println("g anti-commutes with a stabilizer: $(state.tab[:,noncommuting[1]])")
        #the measurement is not compatible with the stabilizer group. the outcome is random
        pivotIdx=noncommuting[1]
        for j in noncommuting[2:end]
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end
        
        #update the out-of-group stabilizers, so tehy commute with the newly added stabilizer
        for j in findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
            addCols(state,pivotIdx,j)
#             the following line is not necassary because the destabilizer for pivotIdx is going to be descarded
#             addCols(state,j+n,pivotIdx+n)     
        end

#         since it is forced measurement
#         outcome=rand([0x00,0x02])
        outcome=0x00
        
        insertStabilizer(state,g,(outcome+gphase)%0x04,pivotIdx)
        updateDestabilizers(state,pivotIdx)
        return outcome
    end
end

#-------------------------------------------------
#-------------------------------------------------       
function totallyMixed(n)
    state=StateDensityMatrix(Matrix{UInt8}(I,2n,2n),zeros(UInt8,2n),0)
    return state
end
#-------------------------------------------------
#-------------------------------------------------          
function xProductState(n)
    state=StateDensityMatrix(Matrix{UInt8}(I,2n,2n),zeros(UInt8,2n),n)
    return state
end
#-------------------------------------------------
#-------------------------------------------------
function zProductState(n)
    state=StateDensityMatrix(zeros(2n,2n),zeros(UInt8,2n),n)
    state.tab[n+1:2n,1:n]=Matrix{UInt8}(I,n,n)
    state.tab[1:n,1+n:2n]=Matrix{UInt8}(I,n,n)
    return state
end
#-------------------------------------------------
#-------------------------------------------------

function applyUnitary(pauliStr,u,uphase)
#     applies unitary gate "u" to pauli string pauliStr
#     u is assumed to be given in the following form. say pauliStr is defined on k qubits
#     i.e. u acts on k qubits. Then u is a 2k x 2k bit matrix where the first
#     k columns are images of X1,...,Xk in the form of Pauli strings (bit 
#     strings of length 2k) and the rest of the columns (k+1 to 2k) are 
#     the images of Z1,...,Zk.
    
    is=findall(isequal(0x01),pauliStr)
    if length(is)==0 return pauliStr, 0x00 end

    img=u[:,is[1]]
    phase=uphase[is[1]]

    for i in is[2:end]
        phase+=uphase[i]+ productSign(img,u[:,i])
        img.⊻=u[:,i]
    end
    
    return img, phase%0x04
end

#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------

function applyUnitary(state,u,uphase,indices)
#     applies unitary gate "u" to qubits with indexs given by "indices"
#     u is assumed to be given in the following form. say length indices =k,
#     i.e. u acts on k qubits. Then u is a 2k x 2k bit matrix where the first
#     k columns are images of X1,...,Xk in the form of Pauli strings (bit 
#     strings of length 2k) and the rest of the columns (k+1 to 2k) are 
#     the images of Z1,...,Zk.
    
    n=size(state.tab,1)÷2

    for j in 1:size(state.tab,2)
        is=findall(isequal(0x01),state.tab[[indices;indices.+n],j])
        if length(is)==0 continue end

        img=u[:,is[1]]
        phase=uphase[is[1]]

        for i in is[2:end]
            phase+=uphase[i]+ productSign(img,u[:,i])
            img.⊻=u[:,i]
        end

        state.phases[j]=(state.phases[j]+phase)%0x04
        state.tab[[indices;indices.+n],j]=img
    end
    return
end

#-------------------------------------------------
#-------------------------------------------------

function decohereX(state,i)
    n=size(state.tab,1)÷2
    r=state.rank
    Xi=zeros(UInt8,2n)
    Xi[i]=0x01
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],Xi))
    if length(noncommuting)==0
        return 0x00
    else
        pivotIdx=noncommuting[1]
#         after this for loop, we would have a valid stabilizer tableau where
#         where only one stabilizer anticommutes with X_i
        
        for j in noncommuting[2:end]
            addCols(state,pivotIdx,j)
#             this is needed to keep the stabilizer/destabilizer valid
            addCols(state,j+n,pivotIdx+n)  
        end
        
        swapCols(state,pivotIdx,r)
        swapCols(state,pivotIdx+n,r+n)
        
        state.rank-=1
        return 0x01
    end
end

#-------------------------------------------------
#-------------------------------------------------

function expval(state,g,gphase=0x00, verbose=false)
    verbose && printstyled("\n calculating expectation value of: $g\n";color=:blue)
    verbose && println("state has the following tableau:")
    verbose && display(state.tab)
    verbose && println("with rank:$(state.rank) and phases:$(state.phases)")
    
    n=size(state.tab,1)÷2
    r=state.rank
    noncommuting=findall(x->x==0x01,commute(state.tab[:,1:r],g))

    if length(noncommuting)==0
        verbose && println("no noncommuting in-group stabilizer")
        #check whether g is in the stabilizer group
        
        noncommuting_S=findall(x->x==0x01,commute(state.tab[:,r+1:n],g)).+r
        if length(noncommuting_S)!=0
            verbose && println("a non-commuting out-of-group stabilizer has been found: g is not in the stabilizer group")
            return 0
        end
        verbose && println("no noncommuting out-of-group stabilizer has been found")
        
        noncommuting_D=findall(x->x==0x01,commute(state.tab[:,r+1+n:2n],g)).+(r+n)
        if length(noncommuting_D)!=0
            verbose && println("a non-commuting out-of-group destabilizer has been found: g is not in the stabilizer group")
            return 0
        end
        verbose && println("no noncommuting out-of-group destabilizer.\n g is in the stabilizer group")

        
#         +-g is in the stabilizer group, so the measurement outcome is deterministic and rank remains the same
        noncommuting_DS=findall(x->x==0x01,commute(state.tab[:,1+n:r+n],g))
        if length(noncommuting_DS)==0
            if sum(g)==0
                return 1
            else
                printstyled("ERROR: SOMETHING IS VERY WRONG: g commutes with everything!\n";color = :red)
                println("g:")
                display(g)
                println("tab:")
                display(state.tab)
                println("rank:$(state.rank)")
                return
            end
        end
        ph=state.phases[noncommuting_DS[1]]
        op=state.tab[:,noncommuting_DS[1]]
        
        for i in noncommuting_DS[2:end]
            ph=ph+state.phases[i]+productSign(op,state.tab[:,i])
            op=op.⊻state.tab[:,i]
        end
        ph=ph%0x04
        
        if g!=op
            
            printstyled("ERROR: SOMETHING IS VERY WRONG:\n $g \n is not equal to\n $op\n";color = :red)
            println("g:")
            display(g)
            println("tab:")
            display(state.tab)
            println("rank:$(state.rank)")
        end
        if gphase==ph
            return 1
        elseif gphase==(ph+0x02)%0x04
            return -1
        else
            printstyled("ERROR: SOMETHING IS VERY WRONG. MAYBE g IS NOT HERMITIONA\n";color = :red)
            return 
        end
            
    else
        verbose && println("g anti-commutes with a stabilizer")
        return 0
    end
end

#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
function allCliffordUnitaries_withPhase(n)
    #generates all n-qubit unitaries, as an array of {u,uphase}. u is a 2nx2n matrix,
    #where th j'th column is the image of X_j for j<=n and image of Z_{j-n} for j>n. uphase is a 
    #an array of length 2n where the j'th element is a phase which makes u[:,j] Hermition. 

    # NOTE:  it doesn't generate all possible phases. It just choose a phase that makes the image Hemritian
    #        therfore, the number of gates in this function is exactly the same as the outputs
    #        of  "allCliffordUnitaries" function.
    
    us_withPhase=[]
    us=allCliffordUnitaries(n)
    for u in us
        uphase=zeros(UInt8,2n)
        for j in 1:2n
            if productSign(u[:,j],u[:,j])==0x02
                uphase[j]=0x01
            end
        end
        push!(us_withPhase,(u,uphase))
    end
    return us_withPhase
end
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------

function allCliffordUnitaries(n)

    #generates all n-qubit Clifford unitaries up to phases. It's only viable for n<=3
    #though. after that it gets too large a set.

    #It uses the algorithem that is  described in Appendix A.c of arXiv:1901.08092

    #outputs an array of u, for each u, the j'th column is the image of X_j for 1<=j<=n 
    #and the image of Z_{j-n} for n+1<=j<=2n

    if n==1
        us=[]
        p=allPossibleBitAssignments(4)
        for j in 1:size(p,2)
            if commute(p[1:2,j],p[3:4,j])==0x01
                push!(us,reshape(p[:,j],2,2))
            end
        end
        return us
    else
        us=[]
        ws=allCliffordUnitaries(n-1)
        v=zeros(UInt8,2n,2n)
        v[1,1]=1
        v[n+1,n+1]=1
        p=allPossibleBitAssignments(4n)

        for j in 1:size(p,2)
            x1p=p[1:2n,j]
            z1p=p[2n+1:end,j]
            if commute(x1p,z1p)==0x01
                uprime=extendToUnitary(x1p,z1p)
                for w in ws
                    v[[2:n;n+2:2n],[2:n;n+2:2n]]=w
                    u=(v*uprime).%0x02
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
function extendToUnitary(x1p,z1p)
    #return a unitary which maps X_1 to the given x1p & Z_1 to the given z1p
    #the first n columns are images of X_i's and the rest are images of the Z_i's

    n=length(x1p)÷2

    if commute(x1p,z1p)==0x00
        print("Error: first vecotrs aren't anti-commuting!")
        return nothing
    end

    #we start by a complete basis for Paulis and replace a pair with x1p,z1p.
    #to choose which pair to replace we find a pair that appear non-trivially
    #in x1p & z1p.
    #this makes sure that after replacement xbasis \cup zbasis still forms a complete set for Pauli strings on n qubits.

    #the j'th column represents X_j
    xbasis=vcat(Matrix{UInt8}(1I,n,n),zeros(UInt8,n,n))

    #the j'th column represents Z_j
    zbasis=vcat(zeros(UInt8,n,n),Matrix{UInt8}(1I,n,n))

    dependentindex=findfirst(i->commute(x1p[[i,i+n]],z1p[[i,i+n]])==0x01,1:n)
    xbasis[:,dependentindex]=x1p
    zbasis[:,dependentindex]=z1p
    z2swapcol!(xbasis,1,dependentindex)
    z2swapcol!(zbasis,1,dependentindex)


    #we now update other basises accordingly, such that [xi,xj]=[zi,zj]=0
    #and {xi,zj}=0

    for i=2:n

        #finding Zi, assuming the commutation relations are satisfied by
        #xj,zj pairs for j<i.


        #make sure Zi commutes with Xj for j<i
        for j=1:i-1
            if commute(zbasis[:,i],xbasis[:,j])==0x01
                zbasis[:,i].⊻=zbasis[:,j]
            end
        end

        #make sure Zi commutes with Zj for j<i
        for j=1:i-1
            if commute(zbasis[:,i],zbasis[:,j])==0x01
                zbasis[:,i].⊻=xbasis[:,j]
            end
        end


        #find an Xj in the remaining baisis vectors that doesn't commute with Zi
        #and name it Xi
        #first search among the remaing Xs

        foundaPair=false
        for j=i:n
            if commute(zbasis[:,i],xbasis[:,j])==0x01
                z2swapcol!(xbasis,j,i)
                foundaPair=true
                break
            end
        end
        #if not found, search among the remaing Zs
        if foundaPair==false
            for j=i+1:n
                if commute(zbasis[:,i],zbasis[:,j])==0x01
                    #swap the two
                    zbasis[:,j].⊻=@view xbasis[:,i]
                    xbasis[:,i].⊻=@view zbasis[:,j]
                    zbasis[:,j].⊻=@view xbasis[:,i]
                    foundaPair=true
                    break
                end
            end
        end

        if foundaPair==false
            print("Error")
            return nothing
        end

        #make sure Xi commutes with Xj for j<i
        for j=1:i-1
            if commute(xbasis[:,i],xbasis[:,j])==0x01
                xbasis[:,i].⊻=zbasis[:,j]
            end
        end

        #make sure Xi commutes with Zj for j<i
        for j=1:i-1
            if commute(xbasis[:,i],zbasis[:,j])==0x01
                xbasis[:,i].⊻=xbasis[:,j]
            end
        end

    end

    return hcat(xbasis,zbasis)
end

#-------------------------------------------------
#-------------------------------------------------
function randomCliffordUnitary(n)

    #generates a random n-qubit Clifford unitary 
    #It uses the algorithem that is  described in Appendix A.c of arXiv:1901.08092
    #outputs a 2nx2n matrix where the j'th column is the image of X_j for 1<=j<=n 
    #and the image of Z_{j-n} for n+1<=j<=2n

    if n==1
        return rand(allCliffordUnitaries(1))
    else
        w=randomCliffordUnitary(n-1)
        v=zeros(UInt8,2n,2n)
        v[1,1]=1
        v[n+1,n+1]=1


        x1p=rand([0x00,0x01],2n)
        z1p=rand([0x00,0x01],2n)
        
        while commute(x1p,z1p)==0x00
            x1p=rand([0x00,0x01],2n)
            z1p=rand([0x00,0x01],2n)
        end

        uprime=extendToUnitary(x1p,z1p)
        
        v[[2:n;n+2:2n],[2:n;n+2:2n]]=w
        u=(v*uprime).%0x02

        return u
    end
    return
end
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
function randomCliffordUnitary_withphase(n)
    #generates a n-qubit random unitaries, with a compatible phase such that images of each Pauli is
    #Hermition (but the phase is not randomized). u is a 2nx2n matrix,
    #where th j'th column is the image of X_j for j<=n and image of Z_{j-n} for j>n. uphase is a 
    #an array of length 2n where the j'th element is a phase which makes u[:,j] Hermition. 
    u=randomCliffordUnitary(n)

    uphase=zeros(UInt8,2n)
    for j in 1:2n
        if productSign(u[:,j],u[:,j])==0x02
            uphase[j]=0x01
        end
    end
    return (u,uphase)
end
#-------------------------------------------------
#-------------------------------------------------

function diagonalizePaulis(paulis)
    # The input is an array of Pauli strings
    # It outputs an equavalent set of paulis,
    # couple_stabs, couple_destabs and sinlge_stabs
    # such that elements of coupl_stabs and couple_destabs are 
    # conjugate of each other (i'th element of one with the i'th element of the other)
    # and elements of single_stabs commute with everything (do not have conjugate in paulis)


    M=length(paulis)
    if M==0
        return
    end
    n=size(paulis[1],1)÷2

    ptab=zeros(UInt8,2n,M)
    for m in 1:M
        ptab[:,m]=paulis[m]
    end


    couple_stabs=[]
    couple_destabs=[]
    sinlge_stabs=[]

    i=1
    while i<=M
        noncommuting=findall(isequal(0x01),commute(ptab[:,i+1:M],ptab[:,i])).+i
        if length(noncommuting)==0
            push!(sinlge_stabs,ptab[:,i])
            i+=1
        else
            pvt=noncommuting[1]
            for j in noncommuting[2:end]
                ptab[:,j]=(ptab[:,j].+ptab[:,pvt]).%2
            end
            swapCols(ptab,pvt,i+1)
            noncommuting_d=findall(isequal(0x01),commute(ptab[:,i+2:M],ptab[:,i+1])).+(i+1)
            if length(noncommuting_d)>0
                for j in noncommuting_d
                    ptab[:,j]=(ptab[:,j].+ptab[:,i]).%2
                end
            end
            push!(couple_stabs,ptab[:,i])
            push!(couple_destabs,ptab[:,i+1])
            i=i+2
        end
    end


    return couple_stabs, couple_destabs, sinlge_stabs
end

#-------------------------------------------------
#-------------------------------------------------

function extendToFullDestab(couple_stabs,couple_destabs,single_stabs)
    # outputs a full destabilizer state whose first stabs (destabs) are 
    # couple_stabs (couple_destabs) and then the next stabs are the single_stabs
    
    M=length(couple_stabs)
    M_s=length(single_stabs)

    if M!=0
        n=length(couple_stabs[1])÷2
    elseif M_s!=0
        n=length(single_stabs[1])÷2
    else
        return -1
    end
    
    state=zProductState(n)
    for m in 1:M
        forceMeasure(state,couple_destabs[m],0x00)
        state.tab=explicitStab(state.tab,couple_destabs[m],m)
        forceMeasure(state,couple_stabs[m],0x00)

    end
    for m in 1:M_s
        forceMeasure(state,single_stabs[m],0x00)
        state.tab=explicitStab(state.tab,single_stabs[m],M+m)
    end
    return state.tab
end

#-------------------------------------------------
#-------------------------------------------------
function explicitStab(stab,g,k)
    # rewrite stab such that g is the k'th column of tab. 
    # assumes that g is in the stabilizer group generated by stab
    n=size(stab,1)÷2
    dcm=decompose(g,stab)
    if sum(dcm[n+1:2n])!=0
        #g is not in the stabilizer group
        return -1
    end
    idxs=findall(isequal(0x01),dcm)
    pvt=idxs[1]
    for i in idxs[2:end]
        stab[:,pvt].⊻=stab[:,i]
        stab[:,i+n].⊻=stab[:,pvt+n]
    end

    swapCols(stab,k,pvt)
    swapCols(stab,k+n,pvt+n)

    return stab
end
#-------------------------------------------------
#-------------------------------------------------

function randomizeDestabforSingles(tab,m_couple,m_single)
    # outputs a full destabilizer state whose first stabs (destabs) are 
    # couple_stabs (couple_destabs) and then the next stabs are the single_stabs
    # whose conjugate destabs are randomized

    n=size(tab,1)÷2

    for m in m_couple+1:m_couple+m_single
        if rand()<1/2
            addCols(tab,m,m+n)
        end
        for mm in m+1:m_couple+m_single
            if rand()<1/2
                addCols(tab,mm,m+n)
                addCols(tab,m,mm+n)
            end
        end
        for mm in m_couple+m_single+1:n
            coin=rand()
            if coin<1/4
                addCols(tab,mm,m+n)
                addCols(tab,m,mm+n)
            elseif coin<2/4
                addCols(tab,mm+n,m+n)
                addCols(tab,m,mm)
            elseif coin<3/4
                addCols(tab,mm,m+n)
                addCols(tab,mm+n,m+n)
                addCols(tab,m,mm)
                addCols(tab,m,mm+n)
            end
        end
    end

    return tab
end

#-------------------------------------------------
#-------------------------------------------------

function decompose(p,basis)
    #find the decomposition of p in terms of the basis in tab
    #assumes basis is a destabilizer/unitary
    n=size(basis,1)÷2
    decomp=commute(basis,p)
    return [decomp[n+1:2n];decomp[1:n]]
end

#-------------------------------------------------
#-------------------------------------------------

function inverseUnitary(tab)
    # returns the inverse of a unitary
    n=size(tab,1)÷2
    return mapslices(x->decompose(x,tab),Matrix{UInt8}(I,2n,2n),dims=[1])
end 

#-------------------------------------------------
#-------------------------------------------------

function randomSymmetricClifford(paulis)
    # paulis is an array pauli strings
    # assumes the columns are linearly independent (Note that it should not includes identity)   

    cs,cds,ss=diagonalizePaulis(paulis)

    
    tab1=extendToFullDestab(cs,cds,ss)
    tab2=randomizeDestabforSingles(copy(tab1),length(cs),length(ss))

    # display(tab1)
    # display(tab2)

    n=size(tab1,1)÷2
    m=length(cs)+length(ss)

    if m<n
        w=Matrix{UInt8}(I,2n,2n)
        w[[m+1:n;m+1+n:2n],[m+1:n;m+1+n:2n]]=randomCliffordUnitary(n-m)

        return (tab2*w*inverseUnitary(tab1)).%0x02
    else
        return (tab2*inverseUnitary(tab1)).%0x02
    end
end

#-------------------------------------------------
#-------------------------------------------------

function randomSymmetricClifford_withphase(paulis)
    # paulis is a 2nxm matrix, where each column is pauli string
    # assumes the columns are linearly independent (Note that it should not includes identity)   
    # also returns a phase vector to make the columns of u Hemritian (does not randomize the phase)

    u=randomSymmetricClifford(paulis)
    n=size(u,1)÷2
    uphase=zeros(UInt8,2n)
    for j in 1:2n
        if productSign(u[:,j],u[:,j])==0x02
            uphase[j]=0x01
        end
    end
    return (u,uphase)
end

#-------------------------------------------------
#-------------------------------------------------

function initGHZ(n)
    #initialize a n-qubit ghz state

    state=StateDensityMatrix(zeros(2n,2n),zeros(UInt8,2n),n)

    #stabs:
    #Z_i Z_i+1
    for i in 1:n-1
        state.tab[i+n:i+1+n,i].=0x01
    end
    #X_1 X_2...X_N
    state.tab[1:n,n].=0x01


    #destabs:
    #X_1....X_i
    for i in 1:n-1
        state.tab[1:i,i+n].=0x01
    end
    #ZN
    state.tab[n+n,n+n]=0x01

    return state
end